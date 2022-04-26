<h1>T-EBAnO - Text-Explaining BlAck-box mOdels</h1>

<p align="center">
    <img src="img/logo.png" alt="EBANO-logo" style="width:200px;"/>
</p>

Explaining predictions of deep learning models for Natural Language Processing tasks.

T-EBAnO is a domain and model specific XAI technique for deep models in the NLP domain.
* **Local Explanations**: explain single predictions of the model.
* **Global Explanations**: aggregate multiple local explanations to globally explain the model.

The methodology is **model-specific**, thus it requires to implement the `model_wrapper_interface` to adapt for your specific model. 
Some examples of interface implementations could be found in the `model_wrappers` folder.

<h2>Survey nPIR correlation</h2>

Link to the <a href="https://docs.google.com/forms/d/e/1FAIpQLSfv6XT0tEjYzVBXJKZSj7RgCIZaEX8NHYbsB8vrTkbMGp-P1w/viewform" target="_blank">Survey</a>

