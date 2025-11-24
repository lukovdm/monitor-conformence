FROM lukovdm/stormvogel:tover-new

RUN apt-get update && apt-get install -y \
    texlive \
    texlive-xetex \
    texlive-science

ARG PYTHON_VERSION
ARG POETRY_VERSION

# Copy the project files into the container
COPY ./pyproject.toml /app/pyproject.toml
COPY ./poetry.lock /app/poetry.lock
COPY ./README.md /app/README.md
WORKDIR /app

# Load existing virtual environment
RUN poetry env use /opt/venv/bin/python

# Install project dependencies
RUN poetry install --without no-docker

COPY . /app

# create /root/.jupyter directory
RUN mkdir -p /root/.jupyter

# Create a random password for the Jupyter Lab
RUN PASSWORD=$(echo -n $(date +%s) | sha1sum | awk '{print $1}') && echo $PASSWORD > /root/jupyter_password.txt

# Set identity provider class to token based
# Set the token to the password
RUN echo "c.NotebookApp.token = '$(cat /root/jupyter_password.txt)'" >> /root/.jupyter/jupyter_notebook_config.py

RUN echo "echo -e '\033[44;37mRun this container with -p 8080:8080 to get access to the Jupyter Lab from your host computer.\033[0m'" >> /root/.bashrc
# Print the Jupyter Lab URL, including the password
RUN echo "echo -e '\033[44;37mJupyter Lab will be running at http://localhost:8080/?token=$(cat /root/jupyter_password.txt) in a minute or so.\033[0m'" >> /root/.bashrc
# Print how to restart this docker instance after leaving it
RUN echo "echo -e \"\033[44;37mTo restart this container, run docker start -i \$(hostname)\033[0m\"" >> /root/.bashrc

CMD ["bash", "-c", "setsid jupyter lab --ip 0.0.0.0 --port=8080 --no-browser --allow-root 0</dev/null > /app/jupyter_lab.log 2>&1 & exec bash"]
