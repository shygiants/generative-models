#!/usr/bin/env bash

PROJECT_NAME=GenModels
CONTAINER_BASENAME=gen-models
DOCKERFILE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPOSITORY="$USER/$CONTAINER_BASENAME"
PORT=""
ARGS=""

# Check command
CMD=$1
shift

if [ ${CMD} = "train" ] || [ ${CMD} = "export" ] || [ ${CMD} = "eval" ] || [ ${CMD} = "serve" ] || [ ${CMD} = "notebook" ]; then
    CONTAINER_NAME="$CONTAINER_BASENAME-${CMD}"

    if [ ${CMD} = "serve" ]; then
        PORT="-p 9000:9000"
    elif [ ${CMD} = "notebook" ]; then
        PORT="-p 8888:8888"
    fi
elif [ ${CMD} = "tensorboard" ]; then
    PORT="-p 6006:6006 -p 6064:6064"
    CONTAINER_NAME="$CONTAINER_BASENAME-tensorboard"
elif [ ${CMD} = "encode" ]; then
    CONTAINER_NAME="$CONTAINER_BASENAME-encode-$1"
elif [ ${CMD} = "build" ]; then
    echo "Only build Docker image."
    JUST_BUILD=YES
else
    echo "Usage: run_docker.sh [train|export|eval|serve|notebook|tensorboard|encode|build]"
    exit 1
fi

POSITIONAL=()
while [[ $# -gt 0 ]]
do
    key="$1"

    case ${key} in
        --no-build)
        NO_BUILD=YES
        shift # past argument
        ;;
        --tensorboard-port)
        shift
        PORT="-p $1:6006 -p 6064:6064"
        shift
        ;;
        *)    # unknown option
        POSITIONAL+=("$1") # save it in an array for later
        shift # past argument
        ;;
    esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters


if [ -z "$NO_BUILD" ]; then
    echo "Building Docker image..."
    if [ -f ${DOCKERFILE_DIR}/build-args.env ]; then
        # build-args.env exists
        ENVS=""
        while read LINE; do
            ENVS="${ENVS} --build-arg ${LINE}"
        done < ${DOCKERFILE_DIR}/proxy.env
    fi

    docker build ${ENVS} -t ${REPOSITORY} ${DOCKERFILE_DIR}
else
    echo "Skip building Docker image."
fi

if ! [ -z "$JUST_BUILD" ]; then
    exit 0
fi

echo "Running \"${CMD}\"..."

# Check environment variables
if [ ! -f ${DOCKERFILE_DIR}/config.sh ]; then
    echo "config.sh not found! Make use of environment variables!"
    if [ -z "$JOB_DIR" ]; then
        UNSET_VARS=" JOB_DIR"
    fi
    if [ -z "$DATASET_DIR" ]; then
        UNSET_VARS="${UNSET_VARS} DATASET_DIR"
    fi
    if [ -z "$NOTEBOOK_PASSWD" ] && [ ${CMD} = "notebook" ]; then
        UNSET_VARS="${UNSET_VARS} NOTEBOOK_PASSWD"
    fi
    if [ ! -z "$UNSET_VARS" ]; then
        echo "Environment variable$UNSET_VARS is not set"
        exit 1
    fi
else
    . ${DOCKERFILE_DIR}/config.sh
fi


# Remove current running container
echo "Stopping the existing container..."
docker stop ${CONTAINER_NAME}
echo "Removing the existing container..."
docker rm ${CONTAINER_NAME}

# Run docker container
docker run \
    -e "PASSWORD=${NOTEBOOK_PASSWD}" \
    --name ${CONTAINER_NAME} \
    ${PORT} \
    -v ${JOB_DIR}:/job-dir -v ${DATASET_DIR}:/dataset \
    -d ${REPOSITORY} ${CMD} $@

docker logs -f ${CONTAINER_NAME}
