COMP6721 – Applied Artificial Intelligence
Summer 2025 – Project Phase II Submission

Project Title: Image Classification using Convolutional Neural Network

Team Name: AI_Bots
Team Members: Shurrab Mohammed (40323793), Oleksandr Yasinovskyy (40241188), Huzaifa Mohammed (40242080)

GitHub Repository: [https://github.com/MHUZAIFA/COMP-6721-Applied-AI]

Instructions to Run the Code:
------------------------------------------
1. Requirements:
	- Python 3.8+
	- torch>=2.0.0
	- torchvision>=0.15.0
	- numpy
	- Pillow
	- scikit-learn
	- matplotlib
	- seaborn
	- tqdm
	- jupyter

   Install dependencies:
   pip install -r requirements.txt


2. Running the Notebook:
Open the Jupyter notebook:

Follow the notebook cells sequentially to adjust the hyperparameters, load the training and test datasets and perform transform operations (data preprocessing), Define the CNN model architecture, train the CNN model, and view evaluation metrics on the test dataset.

3. File Structure:
- `Project_Phase2.ipynb`: Main notebook with code and visualizations.
- `Test` and `Training`: Folder containing preprocessed image features and labels (not included due to size—see GitHub for instructions). 
Note: You should have 'Test' and 'Training' dataset folders should be in the same directory.
- `README.txt`: Instructions and GitHub link.
- `Project_report (AI_Bots).pdf`: Phase II report.

4. Dataset:
Dataset used is from MIT Places2:
http://places2.csail.mit.edu/download.html
(Subset used includes: Museum, Library, Shopping Mall – 5000 images/class)

5. Output:
The notebook generates testing accuracy and the confusion matrix for the CNN model and saves the best model in .pth format.

For questions, contact:
Shurrab Mohammed (40323793)
