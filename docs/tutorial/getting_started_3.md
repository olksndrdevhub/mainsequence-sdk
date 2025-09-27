# Getting Started 3: Orchestration


Now that we have built and test our DataNodes locally. We can start to orchestrate them and  running the Main Sequence
Platform. 


Before anything lets fix our environmetn and commit our changes

1) Compile from your declared deps (most common)
```shell
uv pip compile pyproject.toml --universal -o requirements.txt

```
2) commit all your changes and push. 

2.1)
```shell
git add -A        # stages new, modified, and deleted files
git commit -m "Tutorial files"

```
2.2) Push to the repo

Now this is a little more complicated, as we are working with a security private repository behind the main sequence platform
we need a signed terminal with the correct cryptographic key

use the cli to open a signed terminal

```shell
python -m mainsequence project open-signed-terminal 60

```
```shell
#commit your changes
git push
```


We can do this  in two ways first we will do it manaully


Lets go to our Tutorial Project

https://main-sequence.app/projects/?search=tutorial

the file browser should look something like this

![img.png](project_file_browser.png)

Now click on the scripts folder and select  Create Job + on any of the launchers

and lets call it  Random NMumbe Launcher Manual Job
![img.png](random_number_launcher_create_job.png)

create the job and you should see it under the Jobs tab. As we didnt schedule it nothing has run so far we can run it 
manualy with clicking on the Run  button

you should see a confirmation box on the top right of the screen
![img.png](job_run_confirmation.png)

Now click the Job and you will see all the JobRuns wait a bit for the Job To run 
