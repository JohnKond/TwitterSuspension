Social relations graph mebeddings made by Giannis Kontogiorgakis

Extract multy_relation graph <br/> 
<br/>
> 1st : 23/02 - 23/03 <br/>
  2nd : 23/02 - 23/04 <br/>
  3rd : 23/02 - 23/05 <br/>
  4th : 23/02 - 23/06 <br/>


Plan A
----------------------------------------------------------------------------
Train 1st (70/30 or 80/20) : <br/> 
>Train with cross val on train data , 
keep performance of cross val,
Train model on all (70%) training data,
Predict Test , and keep test performance 
Train model on all data 100%
----------------------------------------------------------------------------
<b>Step 1</b> <br/>
>Train 1st (70/30 or 80/20) -> predict(2nd - 1st) -> Probably not good ?????? <br>
......................................... -> predict(3rd - 1st) -> Probably not good ?????? <br>
......................................... -> predict(4th - 1st) -> Probably not good ?????? <br>
----------------------------------------------------------------------------
<b>Step 2</b> <br/>
>Train 1st (70/30 or 80/20) -> predict(2nd - 1st) -> Probably not good ??????<br>
Train 2nd (70/30 or 80/20) -> predict(3rd - 2nd) -> Probably not good ??????<br>
Train 3rd (70/30 or 80/20) -> predict(4th - 3rd) -> Probably not good ??????<br>
----------------------------------------------------------------------------
<b>Step 3</b> <br>
Automatic pipeline: 
>   -> Auto graph extraction (Ready, need small changes) <br>
    -> Auto graph embedding (Ready, need small changes) <br>
    -> Auto Graph parsing to csv (Ready, need small changes) <br>
    -> Auto finetuning (split, cross validation, testing , ploting .....) <br>

<br>

>Train 1st (70/30 or 80/20) -> Predict Test 1st -> Is good !<br>
Train 2nd (70/30 or 80/20) -> Predict Test 2nd -> Is good !<br>
Train 3rd (70/30 or 80/20) -> Predict Test 3rd -> Is good !<br>
Train 4th (70/30 or 80/20) -> Predict Test 4th -> Is good !<br>
----------------------------------------------------------------------------




# Plan B
If we can train date based on previously trained nodes (to check)
----------------------------------------------------------------------------
<br>
<b> Step 1 </b> 

> Train 1st (70/30 or 80/20) -> predict(2nd - 1st) -> Probably not good ?????? 0.8?  X <br>
........................................ -> predict(3rd - 1st) -> Probably not good ?????? 0.7? X2 <br>
........................................ -> predict(4th - 1st) -> Probably not good ?????? 0.6? X3  <br>
----------------------------------------------------------------------------
<b> Step 2 </b>

> Train 1st (70/30 or 80/20) -> predict(2nd - 1st) -> Probably not good ?????? 0.8? X <br>
Train 2nd (70/30 or 80/20) -> predict(3rd - 2nd) -> Probably not good ?????? 0.78? Y <br>
Train 3rd (70/30 or 80/20) -> predict(4th - 3rd) -> Probably not good ?????? 0.75? Y1 <br>
----------------------------------------------------------------------------
In both steps keep also time spend for all ...