# Federated Learning Project

This project implements a federated learning system where a server coordinates training among multiple clients. Each client contributes its local model updates to the server, which aggregates them to produce a global model.

## Installation

To install the required dependencies for this project, run the following command:


pip install -r requirements.txt

This will install all the necessary packages specified in the `requirements.txt` file.

## Running the Server

To start the server, use the following command:

`python server.py`

The server is configured to run with a minimum of 2 clients and only 2 rounds of training. You can adjust these settings in the `server.py` file if needed.

## Running the Clients

Each client should be run separately to connect to the server. Before running the clients, ensure that the server is already running.

To run a client, use the following command:

`python client1.py`

`python client2.py`


Each client will create a directory called `output` to save the aggregated features received from the server. These features can be used for feature selection or building forecasted models.

## Feature Extraction

The feature extraction piple-line extract all features from each client that will be aggregated in the server 

## Custom Aggregation Strategy

In the `aggStrategy` module, there is a custom aggregation strategy implemented. This strategy is responsible for aggregating the features received from each client. Scripts for additional feature processing or aggregation can be added to this module as needed.
