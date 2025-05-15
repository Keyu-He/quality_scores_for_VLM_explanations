# Study interface for evaluating VLM rationales 

This codebase is for a web interface for our VLM rationales evaluation study. The deployed interface for this codebase can be found [here](https://<link_hidden>/?uid=prolific_random&prolific_queue_name=llava1.5_with_image_q5_i5_s0). Below you will find detailed instructions for setting up the interface and editing the codebase.

Begin by making a fork of this GitHub repo. You can rename the forked repo to whatever you want. Then clone the forked repo to your local machine. You will also need `secret.py` from me, which contains the API key for the logging server.

## Instructions for running the project

There are three main steps to running the project: preparing the data, deploying the project, and approving studies + downloading logs after the study is completed.

### Pt. 1: Preparing the data

1. Place your VLM answer+rationale outputs as a JSON file in the `data` folder. Look at `data/llava1.5_with_image.json` for an example of the format.

2. We will now create a set of user queues, where each queue is a subset of your VLM outputs that will be presented in sequence to the user. Run the following command to prepare the user queues:

    ```
    python src_utils/generate_user_queues.py --data data/<DATA_FILENAME>.json --name <NAME> --num_instances_per_queue <I> --num_queues <Q> --seed <S>
    ```

   This will create `Q` user queues in `web/baked_queues/<NAME>_q<Q>_i<I>_s<S>/` where each queue has `I` instances. The seed `S` is used to sample the instances in the queue. The `NAME` parameter is used to identify the set of user queues (e.g. for the sample data file, an appropriate name might be `llava1.5_with_image`).

### Pt. 2: Deploying the project

1. **Modify package.json**: Go to `package.json`, and modify lines 3 and 25. Line 3 should point to the URL of your project's webpage, which will be `https://<GITHUB_USERNAME>.github.io/REPO_NAME`. Line 25 should point to the URL of the GitHub repo, which will be `https://<GITHUB_USERNAME>.github.io/REPO_NAME`.

2. **Install packages:** Run `npm install`. This will install the packages in `package.json`.

3. **Launch the server locally:** Run `npm run dev`. This will start a local server at `http://localhost:8000/`. 

    By default, the server will load queue data from `web/baked_queues/demo.json`. If you don't have such a file, it will throw an error. That's okay, instead you can load one of the queues that you just created in the data preparation step. 
    
    Instead, you can go to `http://localhost:8000?uid=<QUEUE_NAME>/000`, which will load data for the first queue in the set of user queues created in the directory `web/baked_queues/<QUEUE_NAME>/`. Instead of `000`, you can replace it with any other instance number from `0` to `I-1`, padded with zeros to make it three digits long.

    When you are playing around with the interface locally, you can check the Chrome console (`Inspect Element` -> `Console`) to see the logs that are printed. This will help you debug any issues that you encounter.

4. **Deploy the project:** Run `npm run deploy`. This will build the project and deploy it to GitHub pages.

    URL for deployed project: `https://<GITHUB_USERNAME>.github.io/<REPO_NAME>?uid=prolific_random&prolific_queue_name=<QUEUE_NAME>`. Replace `<QUEUE_NAME>` with the name of one of the sets of user queues that were created in `web/baked_queues`.

    An example of a deployed URL: `https://<link_hidden>/?uid=prolific_random&prolific_queue_name=llava1.5_with_image_q5_i5_s0`

5. **Launch a study on Prolific:** To run a study on prolific, you will need to add some additional fields to the URL:

    - `prolific_id`: the prolific ID of the participant
    - `study_id`: the ID of the specific study you're launching
    - `session_id`: the ID of the session

    Including these fields, the URL for the study that you will submit to Prolific will look like this:

    ```https://<GITHUB_USERNAME>.github.io/<REPO_NAME>/?prolific_id={{%PROLIFIC_PID%}}&study_id={{%STUDY_ID%}}&session_id={{%SESSION_ID%}}&uid=prolific_random&prolific_queue_name=<QUEUE_NAME>```

    If you are piloting the interface, set `prolific_id` to the name of whoever is doing the pilot, the other fields get set to default values.

### Pt. 3: Downloading the logs

After the study is completed, you need to download the logs, approve the user studies and pay out bonuses. I have written a script that downloads the logs for you, and allows you to go through each user one-by-one and decide to approve/reject their submission and pay out the bonus if approved. You can use the following command:

```
python src_utils/download_approve_and_pay_bonuses.py --study_id <STUDY_ID> --study_name <STUDY_NAME>
```

The `STUDY_ID` is whatever ID is assigned by Prolific to the study. The `STUDY_NAME` is a shorthand name of the study, which is used to identify the study in the logs. A format I like to use for `STUDY_NAME` is `<QUEUE_NAME>/<QUEUE_NUMBER>_<NUM_USERS>users`.

There may be some issues with this part of the code, we can debug those together later once you get to this part.

## Codebase setup

The main files you may need to edit are:

- `web/index.html, web/style.css`: The main frontend files. All instructions and interface elements are defined here.

- `src/main.ts`: The main typescript file that controls the flow of the study. It loads the data, initializes the interface, and handles the logic for moving between instances.

- `src/connector.ts`: This file contains the logic for loading the queue data (from `web/baked_queues`) and for logging the user interaction data.

    If you are running the server locally, nothing actually gets logged, but the data that would have been logged is printed to the console. 
    
    If you are running a study with the deployed project, the data will be logged to `https://<link_hidden>.pythonanywhere.com/`. At the end of the study, when you run `src_utils/download_approve_and_pay_bonuses.py`, the data will be downloaded to `/home/shared/vlm_rationales_eval/user_studies_data/`. You can then analyze the code by running the notebook `src_analysis/UserStudyAnalysis.ipynb` -- just set the `setting` variable to the `QUEUE_NAME`.

### Instructions for editing the code

Let me know if you want to edit any part of the main study flow in `main.ts`, but are unsure how to do so. I can help you with that and update the README accordingly.
