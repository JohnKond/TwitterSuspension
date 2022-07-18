#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import logging
import multiprocessing as mp
import queue
import time
import traceback
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.distributed as td
import torch.multiprocessing
from torchbiggraph.distributed import Startable, init_process_group
from torchbiggraph.types import CharTensorType, Rank
from torchbiggraph.util import (
    allocate_shared_tensor,
    split_almost_equally,
    tag_logs_with_process_name,
)


logger = logging.getLogger("torchbiggraph")


################################################################################
# Generic parameter client-server protocol
################################################################################


# FIXME! This will be slow
def _tostring(t: CharTensorType) -> str:
    return "".join(chr(x.item()) for x in t)


def _fromstring(s: str) -> CharTensorType:
    return torch.tensor([ord(x) for x in s], dtype=torch.int8)


STORE_CMD = 1
GET_CMD = 2
JOIN_CMD = 3
SWAP_CMD = 4

_dtypes = [
    # torch.float16,  # Half
    torch.float32,  # Float
    torch.float64,  # Double
    torch.uint8,  # Byte
    # torch.int8,  # Char
    # torch.int16,  # Short
    torch.int32,  # Int
    torch.int64,  # Long
]


class ParameterServer(Startable):
    """
    A simple parameter server. Clients can store tensors, accumulate, and
    get tensors by string key. Operations on the parameter server are globally
    synchronous.

    FIXME: torchbiggraph.rpc should be fixed to not require torch.serialization,
    then most of this code can be removed.
    FIXME: torch.distributed.recv should not require you to provide the
    tensor to write to; the type and size should be sent in the header.
    That would simplify this code a lot.
    """

    def __init__(
        self,
        num_clients: int,
        group_idxs: Optional[Sequence[int]] = None,
        log_stats: bool = False,
    ) -> None:
        self.num_clients = num_clients
        self.parameters: Dict[str, torch.Tensor] = {}
        self.group_idxs = group_idxs
        self.groups: Optional[List["td.ProcessGroup"]] = None
        self.log_stats = log_stats

    def start(self, groups: List["td.ProcessGroup"]) -> None:
        self.groups = (
            [groups[idx] for idx in self.group_idxs]
            if self.group_idxs is not None
            else None
        )
        join_count = 0
        while True:
            # 1. receive the command
            cmd_buffer = torch.full((6,), -1, dtype=torch.long)
            rank = td.recv(cmd_buffer)
            cmd = cmd_buffer[0].item()

            if cmd == STORE_CMD:
                key = self._recv_key(rank, cmd_buffer[1].item())
                self.handle_store(
                    rank,
                    key,
                    cmd_buffer[2].item(),
                    cmd_buffer[3].item(),
                    cmd_buffer[4].item(),
                    cmd_buffer[5].item(),
                )
            elif cmd == GET_CMD:
                key = self._recv_key(rank, cmd_buffer[1].item())
                self.handle_get(rank, key, cmd_buffer[2].item())
            elif cmd == SWAP_CMD:
                key = self._recv_key(rank, cmd_buffer[1].item())
                self.handle_store(
                    rank,
                    key,
                    cmd_buffer[2].item(),
                    cmd_buffer[3].item(),
                    cmd_buffer[4].item(),
                    cmd_buffer[5].item(),
                )
                self.handle_get(rank, key, False)
            elif cmd == JOIN_CMD:
                join_count += 1
                logger.info(f"ParameterServer join: join_count= {join_count}")
                if join_count == self.num_clients:
                    for r in range(self.num_clients):
                        # after sending the join cmd,
                        # each client waits on this ack to know everyone is done
                        # and it's safe to exit
                        td.send(torch.zeros((1,)), dst=r)
                    do_barrier = cmd_buffer[1].item()
                    if do_barrier:
                        logger.info("ParameterServer barrier begin")
                        td.barrier(self.groups[0])
                        logger.info("ParameterServer barrier end")
                    break
            else:
                raise RuntimeError(
                    "Command is unknown value %d from rank %d." % (cmd, rank)
                )

    @staticmethod
    def _recv_key(rank: int, keylen: int) -> str:
        """Receive a string tensor key from a client node."""
        key_buffer = torch.zeros((keylen,), dtype=torch.int8)
        td.recv(key_buffer, src=rank)
        return _tostring(key_buffer)

    def handle_store(
        self, rank: int, key: str, ndim: int, accum: int, overwrite: int, ttype: int
    ) -> None:
        if ndim == -1:
            assert key in self.parameters
            size = self.parameters[key].size()
        else:
            size = torch.empty((ndim,), dtype=torch.long)
            td.recv(size, src=rank)
            size = size.tolist()
        dtype = _dtypes[ttype]
        if not accum and overwrite and key in self.parameters:
            # avoid holding onto 2x the memory
            del self.parameters[key]
        data = torch.empty(size, dtype=dtype)

        start_t = time.monotonic()
        if self.groups is None:
            td.recv(tensor=data, src=rank)
        else:
            outstanding_work = []
            flattened_data = data.flatten()
            flattened_size = flattened_data.shape[0]
            for idx, (pg, slice_) in enumerate(
                zip(
                    self.groups,
                    split_almost_equally(flattened_size, num_parts=len(self.groups)),
                )
            ):
                outstanding_work.append(
                    td.irecv(tensor=flattened_data[slice_], src=rank, group=pg, tag=idx)
                )
            for w in outstanding_work:
                w.wait()
        end_t = time.monotonic()
        if self.log_stats:
            stats_size = data.numel() * data.element_size()
            stats_time = end_t - start_t
            logger.debug(
                f"Received tensor {key} from client {rank}: "
                f"{stats_size:,} bytes "
                f"in {stats_time:,g} seconds "
                f"=> {stats_size / stats_time:,.0f} B/s"
            )

        if accum:
            self.parameters[key] += data
        elif (key not in self.parameters) or overwrite:
            self.parameters[key] = data

    def handle_get(self, rank: int, key: str, send_size: int) -> None:
        if key not in self.parameters:
            assert send_size, "Key %s not found" % key
            td.send(torch.tensor([-1, -1], dtype=torch.long), rank)
            return

        data = self.parameters[key]
        if send_size:
            type_idx = _dtypes.index(data.dtype)
            td.send(torch.tensor([data.ndimension(), type_idx], dtype=torch.long), rank)
            td.send(torch.tensor(list(data.size()), dtype=torch.long), rank)

        start_t = time.monotonic()
        if self.groups is None:
            td.send(data, dst=rank)
        else:
            outstanding_work = []
            flattened_data = data.flatten()
            flattened_size = flattened_data.shape[0]
            for idx, (pg, slice_) in enumerate(
                zip(
                    self.groups,
                    split_almost_equally(flattened_size, num_parts=len(self.groups)),
                )
            ):
                outstanding_work.append(
                    td.isend(tensor=flattened_data[slice_], dst=rank, group=pg, tag=idx)
                )
            for w in outstanding_work:
                w.wait()
        end_t = time.monotonic()
        if self.log_stats:
            stats_size = data.numel() * data.element_size()
            stats_time = end_t - start_t
            logger.debug(
                f"Sent tensor {key} to client {rank}: "
                f"{stats_size:,} bytes "
                f"in {stats_time:,g} seconds "
                f"=> {stats_size / stats_time:,.0f} B/s"
            )


class ParameterClient:
    """Client for ParameterServer.
    Supports store, accumulate, swap, swap-accumulate, and get operations."""

    def __init__(
        self,
        server_rank: int,
        groups: Optional[List["td.ProcessGroup"]] = None,
        log_stats: bool = False,
    ) -> None:
        self.server_rank = server_rank
        self.groups = groups
        self.log_stats = log_stats

    def store(
        self, key: str, src: torch.Tensor, accum: bool = False, overwrite: bool = True
    ) -> None:
        """Store or accumulate a tensor on the server."""
        cmd_rpc = torch.tensor(
            [
                STORE_CMD,
                len(key),
                -1 if accum else src.ndimension(),
                int(accum),
                int(overwrite),
                _dtypes.index(src.dtype),
            ],
            dtype=torch.long,
        )
        td.send(cmd_rpc, self.server_rank)
        td.send(_fromstring(key), self.server_rank)
        if not accum:
            td.send(torch.tensor(list(src.size()), dtype=torch.long), self.server_rank)
        start_t = time.monotonic()
        if self.groups is None:
            td.send(src, dst=self.server_rank)
        else:
            outstanding_work = []
            flattened_src = src.flatten()
            flattened_size = flattened_src.shape[0]
            for idx, (pg, slice_) in enumerate(
                zip(
                    self.groups,
                    split_almost_equally(flattened_size, num_parts=len(self.groups)),
                )
            ):
                outstanding_work.append(
                    td.isend(
                        tensor=flattened_src[slice_],
                        dst=self.server_rank,
                        group=pg,
                        tag=idx,
                    )
                )
            for w in outstanding_work:
                w.wait()
        end_t = time.monotonic()
        if self.log_stats:
            stats_size = src.numel() * src.element_size()
            stats_time = end_t - start_t
            logger.debug(
                f"Sent tensor {key} to server {self.server_rank}: "
                f"{stats_size:,} bytes "
                f"in {stats_time:,g} seconds "
                f"=> {stats_size / stats_time:,.0f} B/s"
            )

    def get(
        self, key: str, dst: Optional[torch.Tensor] = None, shared: bool = False
    ) -> Optional[torch.Tensor]:
        """Get a tensor from the server."""
        cmd_rpc = torch.tensor(
            [GET_CMD, len(key), dst is None, 0, 0, 0], dtype=torch.long
        )
        td.send(cmd_rpc, self.server_rank)
        td.send(_fromstring(key), self.server_rank)
        if dst is None:
            meta = torch.full((2,), -1, dtype=torch.long)
            td.recv(meta, src=self.server_rank)
            ndim, ttype = meta
            if ndim.item() == -1:
                return None
            size = torch.full((ndim.item(),), -1, dtype=torch.long)
            td.recv(size, src=self.server_rank)
            dtype = _dtypes[ttype.item()]
            if shared:
                dst = allocate_shared_tensor(size.tolist(), dtype=dtype)
            else:
                dst = torch.empty(size.tolist(), dtype=dtype)
        start_t = time.monotonic()
        if self.groups is None:
            td.recv(dst, src=self.server_rank)
        else:
            outstanding_work = []
            flattened_dst = dst.flatten()
            flattened_size = flattened_dst.shape[0]
            for idx, (pg, slice_) in enumerate(
                zip(
                    self.groups,
                    split_almost_equally(flattened_size, num_parts=len(self.groups)),
                )
            ):
                outstanding_work.append(
                    td.irecv(
                        tensor=flattened_dst[slice_],
                        src=self.server_rank,
                        group=pg,
                        tag=idx,
                    )
                )
            for w in outstanding_work:
                w.wait()
        end_t = time.monotonic()
        if self.log_stats:
            stats_size = dst.numel() * dst.element_size()
            stats_time = end_t - start_t
            logger.debug(
                f"Received tensor {key} from server {self.server_rank}: "
                f"{stats_size:,} bytes "
                f"in {stats_time:,g} seconds "
                f"=> {stats_size / stats_time:,.0f} B/s"
            )
        return dst

    def swap(
        self,
        key: str,
        src: torch.Tensor,
        dst: Optional[torch.Tensor] = None,
        accum: bool = False,
        overwrite: bool = False,
    ) -> None:
        """Store or accumulate a tensor on the server,
        and then get its current value.
        """
        if dst is None:
            dst = torch.zeros_like(src)

        cmd_rpc = torch.tensor(
            [
                SWAP_CMD,
                len(key),
                -1 if accum else src.ndimension(),
                int(accum),
                int(overwrite),
                _dtypes.index(src.dtype),
            ],
            dtype=torch.long,
        )
        td.send(cmd_rpc, self.server_rank)
        td.send(_fromstring(key), self.server_rank)
        if not accum:
            td.send(torch.tensor(list(src.size()), dtype=torch.long), self.server_rank)
        start_t = time.monotonic()
        td.send(src, self.server_rank)
        td.recv(dst, src=self.server_rank)
        end_t = time.monotonic()
        if self.log_stats:
            stats_size = (
                src.numel() * src.element_size() + dst.numel() * dst.element_size()
            )
            stats_time = end_t - start_t
            logger.debug(
                f"Swapped tensor {key} with server {self.server_rank}: "
                f"{stats_size:,} bytes "
                f"in {stats_time:,g} seconds "
                f"=> {stats_size / stats_time:,.0f} B/s"
            )

    def join(self, do_barrier: bool = False) -> None:
        """All clients should call join at the end, which will allow the server
        to exit.
        If barrier is True, *caller* must actually execute the barrier on
        self.groups[0].
        """
        cmd_rpc = torch.tensor(
            [JOIN_CMD, do_barrier and 1 or 0, 0, 0, 0, 0], dtype=torch.long
        )
        logger.info("ParameterClient join start")
        td.send(cmd_rpc, self.server_rank)
        ack = torch.empty((1,))
        td.recv(ack, src=self.server_rank)
        logger.info("ParameterClient join end")


class GradientParameterClient:
    """We keep track of the last pull of each tensor from the server, and then when
    a push is requested, we accumulate the difference between the pulled tensor
    and the current version
    """

    def __init__(self, server_rank: Rank) -> None:
        self._client = ParameterClient(server_rank)
        self._cache: Dict[str, torch.Tensor] = {}

    def push(self, key: str, tensor: torch.Tensor) -> None:
        # if they tensor is cached, accumulate the difference.
        # otherwise, send the tensor to the server.
        if key in self._cache:
            diff = tensor - self._cache[key]
            self._client.store(key, diff, accum=True)
            self._cache[key] += diff
        else:
            self._cache[key] = tensor.clone()
            # all the clients race to set the initial value of the tensor, then
            # every clients just uses that one
            self._client.store(key, self._cache[key], overwrite=False)

    def pull(self, key: str, dst: torch.Tensor) -> torch.Tensor:
        self._client.get(key, dst)
        if key in self._cache:
            self._cache[key].copy_(dst)
        else:
            self._cache[key] = dst.clone()
        return dst

    def update(self, key: str, tensor: torch.Tensor) -> None:
        if key in self._cache:
            diff = tensor - self._cache[key]
            self._client.swap(key, diff, self._cache[key], accum=True)
            tensor.copy_(self._cache[key])
        else:
            self._cache[key] = tensor.clone()
            # all the clients race to set the initial value of the tensor, then
            # every clients just uses that one
            self._client.swap(key, self._cache[key], self._cache[key], overwrite=False)
            tensor.copy_(self._cache[key])

    def join(self) -> None:
        self._client.join()


################################################################################
# Parameter sharer
################################################################################


MIN_BYTES_TO_SHARD = 1e7  # only shard parameters above 10MB


def _client_thread_loop(  # noqa
    process_name: str,
    client_rank: Rank,
    all_server_ranks: List[Rank],
    q: mp.Queue,
    errq: mp.Queue,
    init_method: Optional[str],
    world_size: int,
    groups: List[List[Rank]],
    subprocess_init: Optional[Callable[[], None]] = None,
    max_bandwidth: float = 1e8,
    min_sleep_time: float = 0.01,
) -> None:
    try:
        tag_logs_with_process_name(process_name)
        if subprocess_init is not None:
            subprocess_init()
        init_process_group(
            rank=client_rank,
            init_method=init_method,
            world_size=world_size,
            groups=groups,
        )

        params = {}
        clients = [
            GradientParameterClient(server_rank) for server_rank in all_server_ranks
        ]
        log_time, log_rounds, log_bytes = time.perf_counter(), 0, 0

        # thread loop:
        # 1. check for a command from the main process
        # 2. update (push and pull) each parameter in my list of parameters
        # 3. if we're going to fast, sleep for a while
        while True:
            tic = time.perf_counter()
            bytes_transferred = 0
            try:
                data = q.get(timeout=0.01)
                cmd, args = data
                if cmd == "params":
                    params[args[0]] = args[1]
                    log_time, log_rounds, log_bytes = time.perf_counter(), 0, 0
                elif cmd == "join":
                    for client in clients:
                        client.join()
                    break
            except queue.Empty:
                pass

            for k, v in params.items():
                param_size = v.numel() * v.element_size()
                bytes_transferred += param_size
                if param_size > MIN_BYTES_TO_SHARD:
                    chunks = v.chunk(len(clients), dim=0)
                    for client, chunk in zip(clients, chunks):
                        client.update(k, chunk)
                else:
                    client_idx = hash(k) % len(clients)
                    clients[client_idx].update(k, v)

            log_bytes += bytes_transferred
            log_rounds += 1
            log_delta = time.perf_counter() - log_time
            if params and log_delta > 60:
                logger.info(
                    f"Parameter client synced {log_rounds} rounds {log_bytes / 1e9:g} "
                    f"GB in {log_delta:g} s ({log_delta / log_rounds:g} s/round, "
                    f"{log_bytes / log_delta / 1e9:g} GB/s)"
                )
                log_time, log_rounds, log_bytes = time.perf_counter(), 0, 0

            comm_time = time.perf_counter() - tic
            sleep_time = max(
                bytes_transferred / max_bandwidth - comm_time, min_sleep_time
            )
            time.sleep(sleep_time)

    except BaseException as e:
        traceback.print_exc()
        errq.put(e)
        raise


class ParameterSharer:
    """Wrapper object that creates a thread, that pulls parameters and pushes
    gradients, in a loop.
    """

    def __init__(
        self,
        process_name: str,
        client_rank: Rank,
        all_server_ranks: List[Rank],
        init_method: Optional[str],
        world_size: int,
        groups: List[List[Rank]],
        subprocess_init: Optional[Callable[[], None]] = None,
    ) -> None:
        self.q = mp.get_context("spawn").Queue()
        self.errq = mp.get_context("spawn").Queue()
        self.p = mp.get_context("spawn").Process(
            name=process_name,
            target=_client_thread_loop,
            args=(
                process_name,
                client_rank,
                all_server_ranks,
                self.q,
                self.errq,
                init_method,
                world_size,
                groups,
                subprocess_init,
            ),
        )
        self.p.daemon = True
        self.p.start()

    def set_param(self, k: str, v: torch.Tensor) -> None:
        self.check()
        self.q.put(("params", (k, v)))

    def check(self) -> None:
        if not self.errq.empty():
            raise self.errq.get()

    def join(self) -> None:
        self.check()
        self.q.put(("join", None))
        self.check()
        self.p.join()
