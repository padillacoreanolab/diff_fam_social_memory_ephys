# diff_fam_social_memory_ephys


.vscode settings: 

changed through the settings tab under workspace we changed the following setting (File>Preference>Settings or Ctrl+,)

Jupyter: Notebook File Root
${workspaceFolder}

this means that the parent folder opened in VSCode is the workspaceFolder which is set as the working directory 

**note: where a jupyternotebook lives is not necessarily where
it is being run, those are two different things. 

- This is important because it is brittle to have imports be relative to where the jupyter notebook file is stored for importing local modules and extremely important for [pickling and unpickling](https://stackoverflow.com/a/2121918). Eg, if you wanted to move notebooks, or have notebooks be portable from computer to computer