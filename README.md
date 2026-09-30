# Deepfake-detection
Deepfake detection project for IKT213 - Machine vision

a photo editor integrating deepfake detection models and image analysis tools.

I tried to make it as clean and organized as possible, lmk if you guys wanna change stuff

actually i just realized maybe you guys know how to do all of this but i personnaly dont O_O

## Quickstart and setup
each team member have to setup their isolated local environment using the following steps:

### 1. Clone the repository
```bash
git clone https://github.com/J3NJ4K/deepfake-detection.git
cd deepfake-detection
```
### 2. Create and activate the repository
so that we use the same library and stuff, and that it works the same for everybody

i think you have to do this where you saved the repository on your pc, like the path or whatever
- windows
```bash
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```
- macOS/linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install project dependecies
idk if it has everything we need but we can still add them further on
```bash
pip install -r requirements.txt
```
### AI models & heavy weights
large binary weight files are excluded from git via .gitignore to prevent repository bloat
- where to store weights:
all downloaded model weights must be placed locally inside the models/ directory
- download link:
uhhh we'll add one later i guess

### Project architecture
the codebase uses a modular layoutu to support parallel development without merge conflicts:
- core/ : image processing
- ui/ : pyside graphical user interface
- detection/ : ai pipeline
- models/ : local directory for model wiehgt files (ignoed by git)
- main.py : main entry point to launch the app

### Git workflow
1. dont commit directly to main
2. create dedicated feature branch for each task : 

we'll figure this out together ig
```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

3. commit and push your branch:
```bash
git add .
git commit -m "feat : description of your changes"
git push -u origin feature/your-feature-name
```
4. code review : open a pull request on github, at least one of us must review and approve the changes before merging into main.