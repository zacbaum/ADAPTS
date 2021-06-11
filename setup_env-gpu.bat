call %UserProfile%\Anaconda3\Scripts\activate.bat

call conda create --name adapts python=3.6.7
call conda activate adapts

call pip install mss
call pip install opencv-python
call conda install cudatoolkit=10.1 cudnn7.6.5
call tensorflow-gpu==2.3.1
