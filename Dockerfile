# This is a dockerfile with jupyter 
# author: Jiayi Cox
FROM continuumio/miniconda3

# specify docker workdir
WORKDIR /home/esol

# copy necessary files from host to docker
COPY --chown=docker:docker environment.yml .

#add customized source code
COPY --chown=docker:docker src_esol src_esol
COPY --chown=docker:docker notebooks notebooks

# install env
RUN conda env create -f environment.yml 

# expose ports
EXPOSE 8888

# activate the new environemtn
SHELL ["conda", "run", "-n", "esol_graph", "/bin/bash", "-c"]

#install customized package, one doesn't need to run python src_esol/setup.py install b/c that gives the same effect of adding customized python package to PYTHONPATH
ENV PYTHONPATH /home/esol/src_esol 

# install ipykernel for jupyter 
RUN python -m ipykernel install --user --name esol_graph


#run jupyter
CMD ["conda", "run", "--no-capture-output", "-n", "esol_graph", "jupyter", "lab", "--allow-root", "--ip='*'"]

