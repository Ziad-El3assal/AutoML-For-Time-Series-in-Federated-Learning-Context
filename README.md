# AutoML-For-Time-Series-in-Federated-Learning-Context

<pre>Federated learning (FL) has emerged as a promising technique for
training machine learning models on decentralized data. It allows
training on data residing on user devices, addressing privacy
concerns and data security limitations. However, selecting the
optimal model architecture and hyperparameters for each FL task
can be a significant challenge.
</pre>

• Why Federated Learning?

    Traditional machine learning often requires
    centralizing data, raising privacy and security
    concerns. Federated learning offers a solution by
    training models on distributed data sets, eliminating
    the need for data transfer. This is particularly
    advantageous in scenarios where:
        - User data is privacy-sensitive (e.g., healthcare,finance).
        - Data is geographically dispersed acrossdevices.
        - Centralized data storage is impractical or infeasible.


• Project Objectives:

    This project proposes a novel AutoML framework for
    federated learning, aiming to automate the process
    of selecting machine learning algorithms and
    optimizing hyperparameters. Our framework will:
        ▪ Leverage meta-learning for efficient algorithm selection on the central server.
        ▪ Perform hyperparameter tuning on each client device to account for local data variations.
        ▪ Aggregate optimized hyperparameters from clients to the server for improved model performance.

### You can run the server using the following command
- Rememeber to rename the environment to `flowerTurtorial` to be updated -before puplishing-
```bash
run.bat number_clients Dataset_path
```

## Building The Knowledge Base
now to build the knowledge base all what you need:
- Put all the datasets u want to run the training on on a directory.
- Choose the models u want to train at the head of the run.py file -Make sure  it's compatible with the ModelEnum Class.
- run the run.py file in power shell or and command prompt except CMD -very important the script depends on closing all the running cmds.
    
    ```
    python run.py <Dataset_dir> <number_of_client>
    ```
`There might be some data that is not compatoble with the code -there ara a lot of variation- so if an error occured in one of the clients terminal all what you need to do  is t o close them all manuallly and it will continue.`

