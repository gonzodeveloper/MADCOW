# MADCOW
## Multivariate Anomaly Detection for Cost Optimized Windfarms

MADCOW is a model and utility set for the detection of sensor anomalies in turbines and wind farms. The utilities module allows us to generate, visualize and load simulated SCADA streams using OpenFAST and FLORIS. We are using a sequential autoencoder as the primary model for anomaly detection. By training an encoder on controlled synthetic data, we can detect anomalies in real time by observing the reconstruction error in the model.


![Model Architecture]({{https://raw.githubusercontent.com/gonzodeveloper/MADCOW/master/docs/model_architecture.png}})

**** STAUS: Work in progress...