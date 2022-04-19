#!/bin/bash
source .dockerenv

it=''
ssh_params=''
gpu_params=''
jupy_params=''
home2='data'
shared_data_folder_params='-v /data/shared/data:/shared_data'

print_usage() {
  echo "The purpose of this script is to run docker container with the proper parameters. The script has the following arguments:"
  echo "The script has the following arguments: "
  echo "     -e   used on ELTE home2 := home"
  echo "          Example usage: docker_run.sh -c /bin/bash -i -e"
  echo "     -c   the command to run inside the container"
  echo "          Example usage: docker_run.sh -c /bin/bash -i"
  echo "                         docker_run.sh -c python script.py -py_param"
  echo "     -p   if given, the container will start an ssh server and map port number 22 from inside to the specified port number outside"
  echo "          Example usage: docker_run.sh -p 2233 -c /bin/bash -i"
  echo "                         After this command if you ssh into the host computer with port number 2233, you find yourself inside the docker"
  echo "                         It is useful for ssh interpreters"
  echo "     -g   Which GPU(s) to use, it will set the CUDA_VISIBLE_DEVICES env variable inside the container"
  echo "          Example usage: docker_run.sh -c /bin/bash -g 2 -i"
  echo "                         docker_run.sh -c python script.py -py_param -g 2"
  echo "     -i   If give, the docker will be run with -it parameter"
  echo "          If you want to attach to the container, specify this option"
  echo "          Example usage: docker_run.sh -p 2233 -c /bin/bash -i"
}

while getopts eihp:j:g:c: flag
do
    # shellcheck disable=SC2220
    case "${flag}" in
        e) home2='home'
           shared_data_folder_params='';;
        c) command=${OPTARG};;
        g) gpu_params="-e CUDA_VISIBLE_DEVICES=${OPTARG}";;
        i) it="-it";;
        p) ssh_params="-v `pwd`/entry_ssh.sh:/entry.sh -p ${OPTARG}:22";;
        j) jupy_params="-v /home/${USER}/.local/:/home/${USER}/.local -p ${OPTARG}:8888";;
        h) print_usage
           exit 0 ;;
    esac
done

mkdir -p /home/${USER}/.pycharm_helpers
mkdir -p /home/${USER}/.cache
mkdir -p /${home2}/${USER}/storage/data
mkdir -p /${home2}/${USER}/storage/${REPO_NAME}/logs
mkdir -p /${home2}/${USER}/storage/${REPO_NAME}/keops_build
mkdir -p /home/${USER}/.config/pudb
touch /${home2}/${USER}/storage/${REPO_NAME}/.bash_history

IMAGE="${IMAGE:-$IMAGE_NAME}"

CONTAINER_ID=$(docker inspect --format="{{.Id}}" ${IMAGE} 2> /dev/null)
if [[ "${CONTAINER_ID}" ]]; then
    docker run --shm-size=50g --rm \
        $it $ssh_params $jupy_params $shared_data_folder_params $gpu_params \
        --gpus all \
        --user $(id -u):$(id -g) \
        -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
        -v `pwd`/../:/workspace \
        -v /${home2}/${USER}/storage:/storage \
        -v /home/${USER}/.pycharm_helpers:/home/${USER}/.pycharm_helpers \
        -v /home/${USER}/.config/pudb:/home/${USER}/.config/pudb \
        -v /home/${USER}/.cache:/home/${USER}/.cache \
        -v /${home2}/${USER}/storage/${REPO_NAME}/.bash_history:/home/${USER}/.bash_history \
        -v /${home2}/${USER}/storage/${REPO_NAME}/keops_build:/libkeops/pykeops/build \
        -v /mnt/g2big/shared:/mnt/g2big/shared \
        -v /etc/sudoers:/etc/sudoers:ro \
        -v /etc/passwd:/etc/passwd:ro \
        -v /etc/group:/etc/group:ro \
        -v /etc/shadow:/etc/shadow:ro \
        --workdir=/workspace \
        $IMAGE $command
else
    echo "Unknown container image: ${IMAGE}"
    exit 1
fi
