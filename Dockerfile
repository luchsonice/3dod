# Base image, have to use the full version to use the git features
FROM python:3.10
# https://huggingface.co/docs/hub/spaces-sdks-docker-first-demo

# RUN apt-get install -y git

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
COPY ./pre-requirements.txt /code/pre-requirements.txt
COPY ./GroundingDINO /code/GroundingDINO
COPY ./sam-hq /code/sam-hq

RUN pip install --no-cache-dir -r /code/pre-requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

CMD ["python", "app.py"]