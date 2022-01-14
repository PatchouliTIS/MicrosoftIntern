@echo OFF
echo.
echo DEPRECATED. Please consider using a direct call to PolarisClient.exe instead.
echo.
echo This batch script has trouble with some combinations of parameters, making it impossible to invoke Polaris in some ways.  Consider using a direct call to PolarisClient.exe instead, avoiding the pitfalls of script-level input string parsing.
@echo ON
set /p IN_TAR_PATH=<%1
shift
set /p IN_QUERY_PATH=<%1
shift
set ALL_OTHER_PARAM=
set USER_NAME=anonymous
set JOB_NAME=anyjob
set SUB_ID=subid

:otherParam
set TEMP_PARAM=%1
if "%TEMP_PARAM%"=="" (
  goto endParam
)
set FLAG=""
set NFLAG=""
set SIDFLAG=""

if "%TEMP_PARAM%"=="Owner" (
  set FLAG=true
  shift
) else if "%TEMP_PARAM%"=="ExperimentID" (
  set NFLAG=true
  shift
) else if "%TEMP_PARAM%"=="NodeID" (
  set SIDFLAG=true
  shift
) else (
  set ALL_OTHER_PARAM=%ALL_OTHER_PARAM% %TEMP_PARAM%
)

if "%FLAG%"=="true" (
  set USER_NAME=%1
)

For /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c-%%a-%%b)
if "%NFLAG%"=="true" (
  set JOB_NAME=%1
)
if "%SIDFLAG%"=="true" (
  set SUB_ID=%1
)

shift

goto otherParam

:endParam

PolarisClient.exe --mp %IN_TAR_PATH% --qp %IN_QUERY_PATH% --user %USER_NAME% --name %JOB_NAME%_%SUB_ID%_%mydate%_%USER_NAME% %ALL_OTHER_PARAM%