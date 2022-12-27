FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu116-py38-torch1120:latest

RUN conda install -n ptca -c conda-forge cudatoolkit=11.6

WORKDIR /libcuann
COPY ./ /libcuann

RUN pip install -r /libcuann/requirements.txt
RUN python /libcuann/setup.py install

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/libcuann/lib/

RUN pip install \
        tqdm==4.56.0 \
        pandas==1.1.5 \
        numpy==1.21.5 \
        msrest==0.6.21 \
        humanfriendly==10.0 \
        loguru==0.6.0 \
        protobuf==3.19.4

RUN pip install azureml-defaults==1.44.0 azureml-dataprep==4.2.2 azureml-dataprep-native==38.0.0 azureml-dataprep-rslex==2.8.1
