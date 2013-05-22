setlocal enabledelayedexpansion
FOR %%i IN (magic haberman letterrecognition landsat ionosphere) DO condor_submit %%i.csub
