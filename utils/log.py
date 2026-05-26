import json
import os
from typing import Any, Optional
from datetime import datetime
from pathlib import Path

class Log:
    persist_log: bool = False
    logname: Optional[str] = None
    job_id: Optional[str] = None
    log_loc: Optional[str] = None
    def __init__(self, log_location: Optional[str] = None):
        self.job_id = None
        self.log_loc = log_location

    @staticmethod
    def log(message: str, level: str = "info", job_id: Optional[str] = None, args: Optional[Any] = None) -> str:
        timestamp = datetime.now().astimezone().isoformat()
        return json.dumps({"level": level, "job_id": job_id, "timestamp": timestamp, "message": message, "args": args})

    def console(self, message: str, args: Optional[Any] = None):
        log = self.log(message, "info", self.job_id, args)
        self.__write(log)
    
    def error(self, message: str, args: Optional[Any] = None):
        log = self.log(message, "error", self.job_id, args)
        self.__write(log)

    def warning(self, message: str, args: Optional[Any] = None):
        log = self.log(message, "warning", self.job_id, args)
        self.__write(log)

    def __write(self, log: str):
        print(log)
        if self.persist:
            logname = str(self.logname) + ".log"
            if os.path.exists(logname):
                with open(logname, "a") as f:
                    f.write(log + "\n")

    def persist(self, log_type: str):
        self.persit = True
        if self.logname is None:
            log_loc = self.log_loc if self.log_loc else "logs/"
            self.logname = log_loc + "/" + log_type + "-" + str(self.job_id) + ".log"
        path = Path(self.logname).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
