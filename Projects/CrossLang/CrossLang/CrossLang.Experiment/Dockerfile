# escape=`
# https://docs.microsoft.com/en-us/virtualization/windowscontainers/quick-start/set-up-environment?tabs=Windows-Server

# BUILD (cd into the output fodler)
# docker build -t myxapproduct2experiment .

# RUN IMAGE (add --name <<name>> is you want to specify the container name)
#   docker run -p 89:89 -p 4024:4024 --hostname=devmachine myxapproduct2experiment
# OR in your output fodler you can start this bat file
#   StartContainer.bat

# RUN as a deamon (add --name <<name>> is you want to specify the container name)
# docker run -i -d -p 89:89 -p 4024:4024 myxapproduct2experiment

# The image will become a running container, list all container with
# docker ps

# Then connect to it like this
# docker exec -it {{ContainerID}} powershell.exe

FROM mcr.microsoft.com/windows:20H2-amd64
# FROM mcr.microsoft.com/dotnet/framework/aspnet:4.8-windowsservercore-ltsc2019

# Default AH structured query port
EXPOSE 89

LABEL "Experiment"="MyExperiment"
LABEL "ExperimentVersion"="1.0"
LABEL "AH"="From Packge"
LABEL "AHVersion"="7.14.0"

WORKDIR c:\payload\

# copy all files for AH
WORKDIR c:\payload\app\
COPY app/. .
WORKDIR c:\payload\
COPY autopilotForImage.ini ./autopilot.ini

# copy all files for Experiment
WORKDIR c:\payload\ApplicationHostData\
COPY ApplicationHostData/. ./
WORKDIR c:\payload\XapSS\
COPY XapSS/. ./

# copy VS remote debugger
WORKDIR C:\remote_debugger
COPY remote_debugger/ .

# Entrypoint to the image
WORKDIR c:\payload\app\
ENTRYPOINT ["StartAH.bat"]