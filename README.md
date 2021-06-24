# ADAPTS
 
## Setup

>Setup is not required on the ADAPTS PC.

*This portion of the guide assumes that Anaconda Navigator has been installed on the PC.*

From within the ADAPTS Project Folder (`...\Documents\ADAPTS`), double click on `setup_env.bat` or `setup_env-gpu.bat` for computers with GPU.

>If you're not sure if you have a GPU, use the non-GPU version. 

Hit `y` to continue when prompted on the screen which will pop-up.

## Usage

### Windows Connect

On the ADAPTS PC, launch the Windows 10 Connect App. 

>Connect should be pinned to the Taskbar on the bottom the screen, but in case it isnot, hit the `Windows Key` and type "Connect" to find the App.

"Snap" the Connect window to the left side of the laptop screen by grabbing the top of the window and dragging the window over to the left side of the screen and then releasing. Alternatively, select the Connect window and press the `Windows Key` + `Left Arrow Key` together.

### Connect Phone to PC

On the Phone, pull down the Notifications Menu from the top of the screen.

On the top of the screen you will see the Wifi, Volume, Bluetooth, and more icons. If you see the "Smart View" icon (Resembles a play button with two curved lines around it), select it. Otherwise, pull this menu down and scroll through the pages until you see "Smart View". 

Select the ADAPTS Laptop to project your screen to. You should now see the phone screen mirrored on the laptop.

### Start ADAPTS

* Use the File Explorer to navigate to `...\Documents\ADAPTS` and double click on `run-adapts.bat`. This will start ADAPTS with Class Activation Mapping overlaid on the current image.

  * Alternatively; use the File Explorer to navigate to `...\Documents\ADAPTS` and double click on `run-adapts-unet.bat`. This will start ADAPTS with UNet Segmentations overlaid on the current image.

* Move the prompt which appears over to the right side of the screen so it does not block the screen-capturing performed by ADAPTS on the mirrored phone image.

* To exit, mouse over the window which pops-up and hit `q`, or close the prompt which appears on screen.

## Accessing Models

To access the model repository if you need to update any models, see [here](https://liveuclac-my.sharepoint.com/:f:/g/personal/rmapzba_ucl_ac_uk/EpD04R0kavRKm-ByMjDYou0B4OnPjdEKSpfYZtcRr_1dVQ?e=icIII8). Then copy the downloaded model(s) into the models folder in ADAPTS.
