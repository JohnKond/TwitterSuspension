Social relations graph mebeddings made by Giannis Kontogiorgakis

Extract multy_relation graph 1st 23/02 - 23/03 
                             2nd 23/02 - 23/04
                             3rd 23/02 - 23/05
                             4th 23/02 - 23/06

----------------------------------------------------------------------------
Train 1st (70/30 or 80/20) -> Train with cross val on train data , 
                                keep performance of cross val,
                                Train model on all (70%) training data,
                                Predict Test , and keep test performance 
                                Train model on all data 100%
----------------------------------------------------------------------------
Step 1 
Train 1st (70/30 or 80/20) -> predict(2nd - 1st) -> Probably not good ??????
.......................... -> predict(3rd - 1st) -> Probably not good ??????
.......................... -> predict(4th - 1st) -> Probably not good ??????
----------------------------------------------------------------------------
Step 2 
Train 1st (70/30 or 80/20) -> predict(2nd - 1st) -> Probably not good ??????
Train 2nd (70/30 or 80/20) -> predict(3rd - 2nd) -> Probably not good ??????
Train 3rd (70/30 or 80/20) -> predict(4th - 3rd) -> Probably not good ??????
----------------------------------------------------------------------------
Step 3
Automatic pipeline: 
    -> Auto graph extraction (Ready, need small changes)
    -> Auto graph embedding (Ready, need small changes)
    -> Auto Graph parsing to csv (Ready, need small changes)
    -> Auto finetuning (split, cross validation, testing , ploting .....)

Train 1st (70/30 or 80/20) -> Predict Test 1st -> Is good !
Train 2nd (70/30 or 80/20) -> Predict Test 2nd -> Is good !
Train 3rd (70/30 or 80/20) -> Predict Test 3rd -> Is good !
Train 4th (70/30 or 80/20) -> Predict Test 4th -> Is good !
----------------------------------------------------------------------------




Plan B
If we can train date based on previously trained nodes.........
----------------------------------------------------------------------------
Step 1 
Train 1st (70/30 or 80/20) -> predict(2nd - 1st) -> Probably not good ?????? 0.8? X
.......................... -> predict(3rd - 1st) -> Probably not good ?????? 0.7? X2
.......................... -> predict(4th - 1st) -> Probably not good ?????? 0.6? X3 
----------------------------------------------------------------------------
Step 2
Train 1st (70/30 or 80/20) -> predict(2nd - 1st) -> Probably not good ?????? 0.8? X
Train 2nd (70/30 or 80/20) -> predict(3rd - 2nd) -> Probably not good ?????? 0.78? Y
Train 3rd (70/30 or 80/20) -> predict(4th - 3rd) -> Probably not good ?????? 0.75? Y1 
----------------------------------------------------------------------------
In both steps keep also time spend for all ...