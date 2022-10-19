# ihs-privacy-paper
Reproducibility for WESAD dataset

Download WESAD data from here: https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download

Unzip the archive and ensure that the `WESAD` folder is placed directly inside the `src` folder of the repository.

To run the experiments, run 
```
python WesadSimulation.py [preprocessing batches] [data proportion]
```
Where:

- `preprocessing batches` is an integer indicating how many parts each participant's preprocessed data should be split into. 10 is a good number. Does not affect final results
- `data proportion` is a float between 0 and 1 indicating how much of the total data to use for the simulation. Lower values mean a faster simulation but more imprecise results.

After the simulation is done, run

```
python ViewResults.py
```

To view the results. This command will produce a file `out_fig.png` that shows training progress for different values of noise scale. This command will also print to console the results of the privacy attack. The `Condition` field shows the noise scale for a particular attack, and `Accuracy` shows the average accuracy for the attacker classifier.

Exact simulation parameters are located in `WesadSimulation.py` and can be modified. For any questions for concerns, please email ajshen@andrew.cmu.edu

