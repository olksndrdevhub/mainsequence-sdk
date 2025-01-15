import os
import yaml
import hashlib
import json
import socket
from mainsequence.tdag.logconf import get_tdag_logger

logger = get_tdag_logger()
#maitenance


def get_host_name():
    return socket.gethostname()

def read_locked_yaml(path):
    from filelock import FileLock
    lock_path=path+".lock"
    lock = FileLock(lock_path)
    with lock:
        with open(path) as stream:
            read = yaml.load(stream, Loader=yaml.UnsafeLoader)
    return read
def write_locked_yaml(path,dict_file):
    from filelock import FileLock
    lock_path = path + ".lock"
    lock = FileLock(lock_path)
    with lock:
        with open(path, 'w') as f:
            data = yaml.dump(dict_file, f, default_flow_style=False, sort_keys=False)

#yaml reading
def read_yaml(path):

    #if not exisit crate
    if not os.path.exists(path):
        empty_yaml={".":"."}
        if not os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path),exist_ok=True)
        write_yaml(path, empty_yaml)

    with open(path) as stream:
        read=yaml.load(stream,Loader=yaml.UnsafeLoader)


    return read

def write_yaml(path, dict_file):
    with open(path, 'w') as f:
        data = yaml.dump(dict_file, f, default_flow_style=False,sort_keys=False)

def write_key_to_yaml (key,key_dict,path):
    try:
        if os.path.exists(path):
            with open(path,"r") as f:
                yaml_file=yaml.load(f,Loader=yaml.UnsafeLoader)
                yaml_file[key] = key_dict
            with open(path, 'w') as f:
                data = yaml.dump(yaml_file, f)
        else:
            with open(path, 'w') as f:
                data = yaml.dump({key:key_dict}, f)
    except Exception as e:
        raise e
        
   

def read_key_from_yaml(key,path):
    yaml_file = read_yaml(path)

    if key in yaml_file:
        return yaml_file[key]
    else:
        return None

def hash_dict(dict_to_hash: dict) -> str:
    dhash = hashlib.md5()
    encoded = json.dumps(dict_to_hash, sort_keys=True,default=str).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


##databases


def copy_drop_database(source_uri:str,target_uri:str,source_container_name:str):
    """
    Copies database from one host to the other and guarantees  that  are no broken time series.
    Parameters
    ----------
    source_uri :
    target_uri :
    source_container_name :

    Returns
    -------

    """
    import subprocess
    import re
    # 1 Drop database in target
    slash = [m.start() for m in re.finditer('/', target_uri)][-1]
    target_host = target_uri[:slash]
    target_db = target_uri[1 + slash:]

    command = f'psql {target_host} -c '
    command = command.split() + [f"drop database {target_db}"]
    logger.debug("dumping target database")
    p = subprocess.Popen(command, stdout=subprocess.PIPE)
    p.wait()

    logger.debug("recreating database")
    command_1 = f'psql {target_host} -c '
    command_1 = command_1.split() + [f"create database {target_db}"]
    p = subprocess.Popen(command_1, stdout=subprocess.PIPE)
    p.wait()
    # 2 copy Database
    logger.debug("copying database")
    tmp_file_path = "data.sql"
    command_dump = f"pg_dump --no-owner --dbname={source_uri} -f {tmp_file_path}"
    command_restore = f"psql {target_uri} -f {tmp_file_path}"
    command_delete_db = f"rm {tmp_file_path}"
    if source_container_name is not None:
        command_dump = f"docker exec {source_container_name}" + " " + command_dump
        command_restore = f"docker exec {source_container_name}" + " " + command_restore

    p = subprocess.Popen(command_dump.split(), stdout=subprocess.PIPE)
    p.wait()
    p = subprocess.Popen(command_restore.split(), stdout=subprocess.PIPE)
    p.wait()
    p = subprocess.Popen(command_delete_db.split(), stdout=subprocess.PIPE)
    p.wait()
    logger.debug("Copy Done")




def read_sql_tmpfile(url,query):
    import tempfile
    import pandas as pd
    import psycopg2
    with tempfile.TemporaryFile() as tmpfile:
        copy_sql = "COPY ({query}) TO STDOUT WITH CSV {head}".format(
           query=query, head="HEADER"
        )
        # conn = db_engine.raw_connection()
        # cur = conn.cursor()
        with  psycopg2.connect(url) as conn:
            # TEMP FOR FUCKED UP BELOW
            # cur = session.connection().connection.cursor()
            cur=conn.cursor()
            cur.copy_expert(copy_sql, tmpfile)
            tmpfile.seek(0)
            df = pd.read_csv(tmpfile, header=0)

        return df
