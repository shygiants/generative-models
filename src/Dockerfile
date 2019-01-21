FROM tensorflow/tensorflow:1.12.0-py3 as base

MAINTAINER Sanghoon Yoon <shygiants@gmail.com>

#####################
# Install libraries #
#####################
RUN apt-get update && apt-get -y install libsm6 libice6 libxrender1 libxext6 libstdc++6

#######################
# Install pip modules #
#######################
FROM tensorflow/tensorflow:1.12.0-py3 as pip

COPY requirements.txt .
RUN pip install -r ./requirements.txt

########
# MAIN #
########
FROM tensorflow/tensorflow:1.12.0-py3

# Copy libraries
COPY --from=base /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu
COPY --from=base /usr/bin /usr/bin

# Copy pip modules
COPY --from=pip /usr/local/lib/python3.5/dist-packages /usr/local/lib/python3.5/dist-packages

# Expose debug port
EXPOSE 6064

# Set working directory
WORKDIR "/tf"

RUN mkdir /dataset && mkdir /job-dir

# Copy all source codes
COPY . .


ENTRYPOINT ["bash", "run.sh"]