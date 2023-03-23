from time import time
import os
import pandas as pd
from utils import utils

class AvgTime():

    INSTANCE = None
    LOGNO = 0

    LOGFOLDER = "performance_utils/logs"
    LOGFILE = "perf_log-%d.csv"

    HISTORYFOLDER = "performance_utils/history"
    HISTORYFILE = "history-%s.csv"

    REMOVE_OLD_LOGS = True
    REFRESH_FREQ = 200

    HISTORYSIZE = 2000

    LOGCOUNT = 0

    NEWRUN = {}

    def __init__(self):
        if AvgTime.INSTANCE == None:
            self.exectime = {}
            self.starttime = {}
            self.endtime ={}
            self.count = {}
            self.history_files = {}
            AvgTime.INSTANCE = self

            utils.rm_dir(AvgTime.LOGFOLDER)
        else:
            pass
        
    def start(self,id):
        if id not in AvgTime.INSTANCE.exectime.keys():
            AvgTime.INSTANCE.exectime[id] = []
        AvgTime.INSTANCE.starttime[id] = time()
        if id not in AvgTime.INSTANCE.endtime.keys():
            AvgTime.INSTANCE.endtime[id] = None
        if id not in AvgTime.INSTANCE.count.keys():
            AvgTime.INSTANCE.count[id] = 0
        AvgTime.INSTANCE.count[id] += 1

    def end(self,id):
        AvgTime.INSTANCE.endtime[id] = time()
        if id not in AvgTime.INSTANCE.starttime.keys():
            raise KeyError("No log with ID %s was found" % id)
        AvgTime.INSTANCE.exectime[id].append(AvgTime.INSTANCE.endtime[id] - AvgTime.INSTANCE.starttime[id])
        

    def summarize(self):
        if AvgTime.REMOVE_OLD_LOGS and utils.path_exists(AvgTime.LOGFOLDER) and AvgTime.LOGCOUNT % AvgTime.REFRESH_FREQ == 0:
            files_to_remove = [os.path.join(AvgTime.LOGFOLDER,f) for f in os.listdir(AvgTime.LOGFOLDER)]
            utils.rm_files(files_to_remove)
            AvgTime.LOGCOUNT = 0

        utils.mkdir(AvgTime.LOGFOLDER)
        utils.mkdir(AvgTime.HISTORYFOLDER)

        filepath = os.path.join(AvgTime.LOGFOLDER, AvgTime.LOGFILE) % AvgTime.LOGNO
        
        dict = {"function call" : [],"count" : [], "total time" : [], "average time" : []}

        for id in AvgTime.INSTANCE.exectime.keys():
            nr = {"function call" : id,"count" : AvgTime.INSTANCE.count[id], "total time" : sum(AvgTime.INSTANCE.exectime[id]), "average time" : sum(AvgTime.INSTANCE.exectime[id]) / AvgTime.INSTANCE.count[id]}
            for k in dict.keys():
                dict[k].append(nr[k])

        df = pd.DataFrame(dict)
        df.to_csv(filepath, index = False)

        if AvgTime.LOGNO < AvgTime.HISTORYSIZE:

            histfilepaths = {s:os.path.join(AvgTime.HISTORYFOLDER, AvgTime.HISTORYFILE) % s for s in AvgTime.INSTANCE.exectime.keys()}

            for id in AvgTime.INSTANCE.exectime.keys():
                try:
                    assert id in AvgTime.INSTANCE.history_files.keys()
                except AssertionError:
                    AvgTime.INSTANCE.history_files[id] = pd.DataFrame()

                try:
                    if AvgTime.INSTANCE.history_files[id].empty:
                        try:
                            AvgTime.INSTANCE.history_files[id] = pd.read_csv(histfilepaths[id])
                        except FileNotFoundError:
                            pass
                except AttributeError:
                    if AvgTime.INSTANCE.history_files[id] == None:
                        try:
                            AvgTime.INSTANCE.history_files[id] = pd.read_csv(histfilepaths[id])
                        except FileNotFoundError:
                            AvgTime.INSTANCE.history_files[id] = pd.DataFrame()

                try:
                    assert id in AvgTime.INSTANCE.NEWRUN.keys()
                except AssertionError:
                    AvgTime.NEWRUN[id] = True

                if AvgTime.NEWRUN[id] == True:
                    if len(AvgTime.INSTANCE.history_files[id].columns) == 0:
                        AvgTime.INSTANCE.current_run = 1
                        AvgTime.NEWRUN[id] = False
                    else:
                        AvgTime.INSTANCE.current_run = int(AvgTime.INSTANCE.history_files[id].columns[-1]) + 1
                        AvgTime.NEWRUN[id] = False

                    # print("DEBUG : current_run :",AvgTime.INSTANCE.current_run)

                AvgTime.INSTANCE.history_files[id].loc[AvgTime.LOGNO,AvgTime.INSTANCE.current_run] = AvgTime.INSTANCE.count[id]

                if AvgTime.INSTANCE.current_run == 1:
                    AvgTime.INSTANCE.history_files[id].index.name = "Summarize Step"
                    AvgTime.INSTANCE.history_files[id].to_csv(histfilepaths[id], index = True)
                else:
                    AvgTime.INSTANCE.history_files[id].loc[AvgTime.LOGNO,"Summarize Step"] = AvgTime.LOGNO
                    AvgTime.INSTANCE.history_files[id].to_csv(histfilepaths[id], index = False)

        AvgTime.LOGNO += 1
        AvgTime.LOGCOUNT += 1


class AvgTimeWrapper(AvgTime):

    def __init__(self,id):
        self.id = id
        super().__init__()


    def __enter__(self):
        super().start(self.id)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        super().end(self.id)