Assignment 4

This repository uses some code for environments and algorithms found in the credits at the bottom. It is worth noting that you need to use the `mdp_copy` file and `modified_frozen_lake.py` file from this repository as I have modified them to suit my needs.

1. Clone the repository: https://github.com/BradHudson/A4/tree/master
2. Ensure you have python 3 installed
3. Overwrite your local frozen_lake.py environment with my `modified_frozen_lake.py` contents. 
4. You will need to install the python libraries matplotlib, numpy, gym, and seaborn
5. To run the forest fire management value iteration experiment, run `python3 singlepatch_vi.py`
6. To run the forest fire management policy iteration experiment, run `python3 singlepatch_pi.py`
7. To run the forest fire management q learning experiment, run `python3 singlepatch_q.py`
8. To run the frozen lake value iteration experiment, run `python3 frozen_lake_vi.py`
9. To run the frozen lake  policy iteration experiment, run `python3 frozen_lake_pi.py`
10. To run the frozen lake q learning experiment, run `python3 frozen_lake_q.py`

Credits:
- Paper on forest fire management: http://www.mssanz.org.au/MODSIM97/Vol%202/Possingham.pdf
- Code for the forest fire management environment: https://gist.github.com/sawcordwell/bccdf42fcc4e024d394b
- Fork of the mdptoolbox: https://github.com/hiive/hiivemdptoolbox
