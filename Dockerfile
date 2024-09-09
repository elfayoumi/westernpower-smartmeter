FROM continuumio/miniconda3
ADD environment.yml /tmp/environment.yml
WORKDIR /tmp
RUN conda env create -f environment.yml
RUN echo "source activate tensorflow_p36" > ~/.bashrc
ENV PATH /opt/conda/envs/tensorflow_p36/bin:$PATH
COPY  inference_server /inference_server
EXPOSE 5557
EXPOSE 80
ENTRYPOINT ["python", "/inference_server/server.py"]