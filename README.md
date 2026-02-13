# S&P 500 Regime Deteciton 
If Federal Reserve Interest rate hikes are the main trigger for switching from Bull to Bear regimes. 

## Getting Started

- Clone the repository locally using git clone https://github.com/oscar10-wq/S-P500_Regime_detection.git.

- Create your virtual environment from the project's root using pipenv install in the terminal

- Sync up the dependencies using pipenv sync.

Note: If you haven't installed pipenv, you may run the following in the terminal: python3 -m pip install pipenv

## Adding/Removing Dependencies

You can add/remove dependencies directly in the Pipfile. Once that is done, run pipenv lock to generate a new Pipfile.lock, you may verify that the new Pipfile.lock is up to date by running pipenv verify. You can then push your updated Pipfile and Pipfile.lock to the remote branch.

Note: Users will need to pull your changes and run pipenv sync for it to carry over in their virtual environments.


