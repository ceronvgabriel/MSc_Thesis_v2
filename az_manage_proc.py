import log

def delete():
    from azureml.core.compute import ComputeTarget, ComputeInstance
    from azureml.core.compute_target import ComputeTargetException
    from azureml.core import Workspace
    import os
    
    sid='e12cdd49-c6fd-4978-bf96-2bfd69dfd037' 
    CI_NAME=os.popen("echo $CI_NAME").read().rstrip() # Strip new line
    WS_NAME="notebooksEast"
    RG_NAME="notebooksEast"
    ws = Workspace.get(name=WS_NAME, subscription_id=sid, resource_group=RG_NAME)

    try:
        instance = ComputeInstance(workspace=ws, name=CI_NAME)
        print('Found existing instance, use it.')
    except ComputeTargetException:
        print("Instance not found")
    instance.delete(wait_for_completion=True, show_output=True)

def delete_instance(instance):
    try:
        instance.delete(wait_for_completion=True, show_output=True)
    except Exception as e:
        print("Serius error deleting: " + str(e))
        log.log("ERROR DELETING: " + str(e))

def check_vm():
    flag=False
    try:
        from azureml.core.compute import ComputeTarget, ComputeInstance
        from azureml.core.compute_target import ComputeTargetException
        from azureml.core import Workspace
        import os
        sid='e12cdd49-c6fd-4978-bf96-2bfd69dfd037' 
        CI_NAME=os.popen("echo $CI_NAME").read().rstrip() # Strip new line
        WS_NAME="notebooksEast"
        RG_NAME="notebooksEast"
        ws = Workspace.get(name=WS_NAME, subscription_id=sid, resource_group=RG_NAME)
        instance = ComputeInstance(workspace=ws, name=CI_NAME)
        print('Found existing instance, use it.')
        flag=True
    except Exception as e:
        print("Error loading vm instance: " + str(e))
        flag= False
    return flag

# Decorators:
# def run_and_delete(func):
#     def wrapper(*args, **kwargs):
#         if check_vm() is not True:
#             raise Exception("Instance not found")
#         log.log("running")
#         try:
#             func(*args, **kwargs)
#         except Exception as e:
#             print("Error: "+str(e))
#         print("Deleting VM")
#         log.log("end run, deleting")
        
#         delete()
#     return wrapper

# def run_and_delete_test(func):
#     def wrapper(*args, **kwargs):
#         if check_vm() is not True:
#             raise Exception("Instance not found")
#         log.log("running")
#         try:
#             func(*args, **kwargs)
#         except Exception as e:
#             print("Error: "+str(e))
#         log.log("end run, deleting")
#         print("Deleting VM (test)")
#         #delete()
#     return wrapper

def run_and_delete(func,*args,**kwargs):
    log.log("Running then deleting")
    if check_vm() is not True:
        raise Exception("Instance not found")
    log.log("Instance Found, delete possible, running: ")
    
    try:
        func(*args, **kwargs)
    except Exception as e:
        print("Error: "+str(e))
        log.log("Error: "+str(e))
    print("Deleting VM")
    log.log("end run, deleting vm")
    delete()

def run_and_delete_test(func,*args,**kwargs):
    log.log("Running then deleting")
    if check_vm() is not True:
        raise Exception("Instance not found")
    log.log("Instance Found, delete possible, running: ")
    
    try:
        func(*args, **kwargs)
    except Exception as e:
        print("Error: "+str(e))
        log.log("Error: "+str(e))
    print("Deleting VM")
    log.log("end run, deleting vm")
    #delete()