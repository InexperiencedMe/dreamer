OLD REPO. Updated, working version is [here](https://github.com/InexperiencedMe/NaturalDreamer)



### How to use

Even during development phase, someone might still try to run the code to test it out, so, here's a rough guideline:

Ensure all libraries are installed. The code might run, but then throw an error while trying to save a file or saving metrics, as they use new libraries.
We will need not only gymnasium for RacingCar, but also opencv-python, plotly, imageio, pandas and obvious things like pytorch.

I believe you'll have to also create folders, that I have locally, but that arent a part of the repo: 'checkpoints', 'metrics', 'plots', 'videos' and optionally 'results'. I'll put them empty in the repo one day maybe.

Then, you can run the main script (currently as jupyter notebook) with 'resume=False' as you dont have any checkpoints yet.

That's mostly it, the rest should be hopefully quite intuitive. Feel free to contact me for any questions and problems.
