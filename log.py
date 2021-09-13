import logging
import time
import json
import time
import os
import sys
import traceback

logging.basicConfig(filename='log.log', level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',datefmt='%Y-%m-%d %H:%M:%S')

def log(message):
    logging.info(message)

def log_tarin(func,ad_info=""):
    def wrapper(*args, **kwargs):
        print()
        start = time.process_time()
        try:
            best_acc=func(*args, **kwargs)
            total=time.process_time() - start
            log("Exec time: " + str(total) + "best_acc: "  + str(best_acc) + " " + ad_info)
        except Exception as e:
            total=time.process_time() - start
            log("Exec time: " + str(total) + " ERROR: " + str(e) + " " + ad_info)
    return wrapper

#generic log, saves returned value
# def log_save(func,name,raise_=True,ad_info=None):
#     if ad_info==None:
#         ad_info=func.__name__
#     def wrapper(*args, **kwargs):
#         start = time.process_time()
#         to_save=[]
#         try:
#             to_save=func(*args, **kwargs)
#             total=time.process_time() - start
#             log("Exec time: " + str(total) +" "+ ad_info)
#         except Exception as e:
#             total=time.process_time() - start
#             log("Exec time: " + str(total) + " ERROR: " + str(e) + " " + ad_info)
#         with open("./results/"+name, 'w+') as fp:
#             json.dump(to_save,fp)
#     return wrapper
#Use to load:
# with open("file.json", 'r') as fh:
#   orig=json.load(fh)


def log_save(name,func,*args, **kwargs):


    if not os.path.exists("./results/" + os.path.dirname(name)):
            os.makedirs("./results/" + os.path.dirname(name), mode=777)

    res=func(*args, **kwargs)

    # if not os.path.isdir('./results/'+name):
    #     os.makedirs('./results/'+name, mode=0777)

    with open("./results/"+name, 'w+') as fp:
        json.dump(res,fp)
    
    log("Result Saved: " + name)
    
    return res

def try_catch(func,*args,**kwargs):
    raise_e=False
    try:
        log("Running")
        res=func(*args, **kwargs)
        log("Exec succed")
    except Exception as e:
        log("Exception: " + " ERROR: " + str(e))
        res=0
        traceback.print_exc(file=sys.stdout)
        with open("log.log","a") as fp:
            traceback.print_exc(file=fp)
        if raise_e:
            raise ValueError('try_catch: Exception: ' + str(e))
    return res

def exception_handler(func):
    def inner_function(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except TypeError:
            print(f"{func.__name__} only takes numbers as the argument")
    return inner_function

def exception_handler_e(func):
    def inner_function(*args, **kwargs):
        try:
            res=func(*args, **kwargs)
            log("Exec succed")
        except Exception as e:
            print(f"{func.__name__} only takes numbers as the argument")
            log("Exception: " + " ERROR: " + str(e))
            res=None
        return res
    return inner_function

def log_time(func,*args,**kwargs):
    ad_info=""
    log("Start, logging Exec time. "+"Function: " + str(func.__name__) + " ad_info: " + ad_info)
    start = time.process_time()
    res = func(*args, **kwargs)
    total=time.process_time() - start
    log("End, Exec time: " + str(total) +" "+" ad_info: " + ad_info)
    return res





#Usage:
#Or use: from functools import partial; to bind function and arguments
# epochs=200
# model_desc="test_stride"
# ad_info=" saved: " + model_desc + " epochs: " + str(epochs)

# log_time(train_model,ad_info=ad_info)(model,epochs,model_desc)

#Train and delete vm
