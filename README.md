# INR-inspector
Spins up a webpage that can be used to explore the weights and activations of an MLP trained to fit an image.

To start, run

```bash
python inspect_siren.py sirens/coin/best_model_23.pt
```

In a web browser, navigate to http://localhost:5214/ and enjoy! (If you're running the inspect_siren server over ssh, you'll need to do ssh port forwarding to access the webpage in a local browser)

For more information, do

```bash
python inspect_siren.py --help
```