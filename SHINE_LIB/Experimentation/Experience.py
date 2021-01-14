
import os
from subprocess import Popen, PIPE
import time
import yaml
from shutil import copyfile

class Experience():

    def __init__(self, dirs = "Experiences", **kwargs):
        self.name = type(self).__name__
        while not os.path.exists(dirs):
            os.makedirs(dirs)

        cpt = 1
        self.dir = os.path.join(dirs, self.name +"_" + str(cpt))


        while os.path.exists(self.dir):
            cpt = cpt + 1 
            self.dir = os.path.join(dirs, self.name +"_" + str(cpt))

        os.makedirs(self.dir)

        print("Started experience ", self.name, " in  file : ",self.dir)
        self.params = kwargs
        self.conf_dir = None
        self.kw = {}
        self.read_conf()

    def read_conf(self,conf_dir = None):
        if(not conf_dir):
            conf_dir = os.path.join("conf",type(self).__name__+".yaml")
            print("seeking for ", conf_dir)
            if not os.path.exists(conf_dir):
                conf_dir = os.path.join("conf","Experience"+".yaml")
        self.conf_dir = conf_dir
        with open(conf_dir) as f:
            docs = yaml.load_all(f, Loader=yaml.FullLoader)
            self.kw.update(list(docs)[0])
        print("reader conf file : ",self.kw)
        return self.kw
    
    def make_resdir(self,**kwargs):
        ch = "RUN_"
        s = sorted(kwargs.items(), key = lambda x:x[0],reverse = True)
        for i,j in kwargs.items():
            ch += str(i) + "_" + str(j) + "_"
        ch = ch [:-1]
        i = 1
        sh = ch + "_" +str(i)
        while os.path.exists(os.path.join(self.dir, sh)):
            sh = ch + "_" +str(i)
            i = i + 1
        os.makedirs(os.path.join(self.dir, sh))
        print("Created resdir : ",os.path.join(self.dir, sh))
        copyfile(self.conf_dir, os.path.join(os.path.join(self.dir,sh), "conf.yaml") )
        return os.path.join(self.dir, sh)

    
    
    def run(self,):
        pass


