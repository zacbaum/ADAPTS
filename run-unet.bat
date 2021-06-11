call %UserProfile%\Anaconda3\Scripts\activate.bat
call conda activate adapts

python %~dp0\adapts.py 1 1 2> %~dp0\err.txt

call conda deactivate
