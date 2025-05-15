from secret import PROLIFIC_API_KEY, PYTHONANYWHERE_KEY
import argparse
import os
import copy
import requests
import pdb
import pandas as pd
import jsonlines, json
from collections import defaultdict, Counter


SAVE_DIR = "/home/shared/vlm_rationales_eval/user_studies_data/"
print(PROLIFIC_API_KEY)
def get_submissions(study_id):
    out = requests.get(
        f"https://api.prolific.com/api/v1/studies/{study_id}/submissions/",
        params={"state": "AWAITING REVIEW"},
        headers={"Authorization": f"Token {PROLIFIC_API_KEY}"}
    )
    if not out.ok:
        exit("Unable to complete an important request (fetching submissions).")
    d = out.json()["results"]
    return d

def display_interaction_summary(uid, interaction_summary):
    print(f"UID: {uid}")
    print(f"\tBalance: ${interaction_summary['balance']:.2f}")
    print(f"\tAnswer only: {interaction_summary['answeronly_summary']['num_unsures']} unsure, {interaction_summary['answeronly_summary']['num_correct']} correct, {interaction_summary['answeronly_summary']['num_incorrect']} incorrect")
    print(f"\tWith explanation: {interaction_summary['withexplanation_summary']['num_unsures']} unsure, {interaction_summary['withexplanation_summary']['num_correct']} correct, {interaction_summary['withexplanation_summary']['num_incorrect']} incorrect")
    print(f"\tWith explanation quality: {interaction_summary['withexplanationquality_summary']['num_unsures']} unsure, {interaction_summary['withexplanationquality_summary']['num_correct']} correct, {interaction_summary['withexplanationquality_summary']['num_incorrect']} incorrect")
    print(f"\tNumber of exits: {interaction_summary['num_exits']}")

def is_repetitive(interactions, threshold=0.9):
    """
    Returns True if the user has selected the same answer for all answer_types
    in more than 'threshold' fraction of interactions.
    """
    answer_types = ['answeronly', 'withexplanation', 'withexplanationquality']
    selections = []
    for answer_type in answer_types:
        print([x['user_selections'][answer_type] for x in interactions if answer_type in x['user_selections']])
        selections.extend([x['user_selections'][answer_type] for x in interactions if answer_type in x['user_selections']])
    if not selections:
        return False
    counts = Counter(selections)
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count >= threshold * len(selections)

def reject_submission(submission_id, participant_id, message, rejection_category):
    if getattr(args, 'dry_run', False):
        print(
            f"(dry-run) Would reject {submission_id} of {participant_id}: {message}"
        )
        return
    
    if rejection_category not in ["TOO_QUICKLY", "TOO_SLOWLY" ,"FAILED_INSTRUCTIONS", "INCOMP_LONGITUDINAL", "FAILED_CHECK" ,"LOW_EFFORT",
                       "MALINGERING" ,"NO_CODE", "BAD_CODE", "NO_DATA", "UNSUPP_DEVICE", "OTHER"]:
        print(rejection_category)
    r = requests.post(
        f"https://api.prolific.com/api/v1/submissions/{submission_id}/transition/",
        headers={"Authorization": f"Token {PROLIFIC_API_KEY}"},

        json={
            "action": "REJECT",
            "message": message,
            "rejection_category": rejection_category
        }
    )
    if not r.ok:
        exit(
            f"Failed rejection of {submission_id} of participant {participant_id}"
        )
    d = r.json()
    print(f'status: {d["status"]}, participant: {d["participant"]}')

def approve_and_pay_bonuses(study_id, session_id, participant_id, bonus_payment):
    r = requests.post(
        f"https://api.prolific.com/api/v1/submissions/{session_id}/transition/",
        headers={"Authorization": f"Token {PROLIFIC_API_KEY}"},
        json={
            "action": "APPROVE",
        }
    )
    if not r.ok:
        pdb.set_trace()
        exit(
            f"Failed approval of {session_id} of participant {participant_id}"
        )
    d = r.json()
    print(f'status: {d["status"]}, participant: {d["participant"]}')

    if float(bonus_payment) <= 0.0:
        print(f"Bonus payment is 0, not setting up bonus payment.")
        return

    bonus_string = f"{participant_id},{bonus_payment}"
    r = requests.post(
        f"https://api.prolific.com/api/v1/submissions/bonus-payments/",
        headers={"Authorization": f"Token {PROLIFIC_API_KEY}"},
        json={
            "study_id": study_id,
            "csv_bonuses": bonus_string
        }
    )
    if not r.ok:
        exit("Failed to set up bonus payment " + r.content.decode())
    d = r.json()
    print("Set up bonus payment with the following response:", d)
    print(f"Bonus paid: {bonus_payment}, total cost: ${d['total_amount']/100:.2f}")
    bonus_id = d["id"]

    r = requests.post(
        f"https://api.prolific.com/api/v1/bulk-bonus-payments/{bonus_id}/pay/",
        headers={"Authorization": f"Token {PROLIFIC_API_KEY}"},
    )
    if not r.ok:
        exit("Failed to pay bonus payment " + r.content.decode())
    d = r.json()
    print("Paid the bonus with the following response:", d)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study_id', type=str, required=True)
    parser.add_argument('--study_name', type=str, required=True)
    args = parser.parse_args()

    interaction_data_filename = f"{SAVE_DIR}/prolific_batches/batch_interaction_data/{args.study_name}.json"
    batch_summaries_filename = f"{SAVE_DIR}/prolific_batches/batch_summaries/{args.study_name}.tsv"
    tmp_filename = "tmp.jsonl"
    command = f"curl \"<link_hidden>/{args.study_id}\" > {tmp_filename}"
    os.system(command)

    submissions = get_submissions(args.study_id)
    submissions.sort(key=lambda x: x['started_at'])
    print(f"Number of Prolific submissions: {len(submissions)}")

    uid2interactions = defaultdict(list)
    all_interactions = []
    with jsonlines.open(tmp_filename) as reader:
        for obj in reader:
            if "PROLIFIC_PID" in obj['url_data']:
                obj['url_data']['prolific_id'] = obj['url_data']['PROLIFIC_PID']
                obj['url_data']['study_id'] = obj['url_data']['STUDY_ID']
                obj['url_data']['session_id'] = obj['url_data']['SESSION_ID']
                del obj['url_data']['PROLIFIC_PID']
                del obj['url_data']['STUDY_ID']
                del obj['url_data']['SESSION_ID']

            user_id = obj['url_data']['prolific_id']
            try:
                assert obj['question_i'] == len(uid2interactions[user_id])
                uid2interactions[user_id].append(obj)
                all_interactions.append(obj)
            except:
                print(f"Skipping question {obj['question_i']} for user {user_id} because it is out of order (already have {len(uid2interactions[user_id])} interactions).")
                continue
    print(f"Loaded interaction data from PythonAnywhere logs for {len(uid2interactions)} users.")
    print("-"*100)
    # pdb.set_trace()

    output_data = {}
    for s in submissions:
        uid = s['participant_id']
        #print(uid)
        if s['status'] == 'RETURNED':
            print(f"UID: {uid} was returned.")
            print("-"*100)
            continue

        try:
            assert uid in uid2interactions
        except:
            print(f"Interaction data not found in PythonAnywhere logs for User {uid}.")
            print('-'*100)
            #pdb.set_trace()
            continue
        if len(uid2interactions[uid]) != 10:
            continue
        
        balance = round(uid2interactions[uid][-1]['balance']['new'], 2)

        def performance_summary(interactions, type):
            num_unsures = len([x for x in interactions if x['user_selections'][type] == 2])
            num_correct = len([x for x in interactions if x['user_is_correct'][type] == 1])
            num_incorrect = len([x for x in interactions if x['user_is_correct'][type] == 0])
            return {
                'num_unsures': num_unsures,
                'num_correct': num_correct,
                'num_incorrect': num_incorrect,
            }
        interactions = uid2interactions[uid]
        answeronly_summary = performance_summary(interactions, 'answeronly')
        withexplanation_summary = performance_summary(interactions, 'withexplanation')
        withexplanationquality_summary = performance_summary(interactions, 'withexplanationquality')
        num_exits = sum([x['count_exited_page'] for x in interactions])
        interaction_summary = {
            'balance': balance,
            'answeronly_summary': answeronly_summary,
            'withexplanation_summary': withexplanation_summary,
            'withexplanationquality_summary': withexplanationquality_summary,
            'num_exits': num_exits
        }

        output_data[uid] = copy.deepcopy(s)
        output_data[uid]['interaction_summary'] = interaction_summary
        output_data[uid]['interactions'] = interactions    
        if s['status'] in ['APPROVED', 'REJECTED']:
            display_interaction_summary(uid, interaction_summary)
            print(f"Already {s['status']}.")

        elif s['status'] == 'RETURNED' or s['status'] == 'ACTIVE':
            continue

        elif s['status'] == 'AWAITING REVIEW' or s['status'] == 'TIMED-OUT':
            print(f"UID: {uid} is {s['status'].lower()}.")
            display_interaction_summary(uid, interaction_summary)
            try:
                assert len(interactions) == 10
            except:
                print(f"Number of interactions is {len(interactions)}")
                pdb.set_trace()
                print("-"*100)
                continue
            
            # Check for repetitive responses (>90% same answer)
            if (is_repetitive(interactions)):
                warning_message = (f"Warning: User {uid} has selected the same answer for >90% of the time. "
                                "Recommend rejecting submission due to low effort.")
                print(warning_message)

            while True:
                choice = input(f"(A)pprove, (R)eject, (S)kip: ")
                if choice.upper() == 'A':
                    while True:
                        pay_bonus = input(f"Award bonus of ${balance:.2f} to {uid}? Y/N: ")
                        if pay_bonus.upper() == 'Y':
                            out = approve_and_pay_bonuses(args.study_id, s['id'], uid, balance)
                            output_data[uid]['status'] = 'APPROVED'
                            break
                        elif pay_bonus.upper() == 'N':
                            break

                    break
                elif choice.upper() == 'R':
                    # Ask for the reason for rejection.
                    reason_choice = input("Is the rejection because of low effort (overly repetitive responses)? (Y/N)(Any other button to cancel): ")
                    if reason_choice.upper() == 'Y':
                        reject_submission(
                            submission_id=s['id'],
                            participant_id=uid,
                            message="User responses are overly repetitive (>90% identical answers).",
                            rejection_category="LOW_EFFORT"
                        )
                        output_data[uid]['status'] = 'REJECTED'
                        break
                    elif reason_choice.upper() == 'N':
                        print("Rejecting submission without reason.")
                        output_data[uid]['status'] = 'REJECTED'
                        break
                    else:
                        print("Cancellation detected. No rejection has been applied.") 
                elif choice.upper() == 'S':
                    break

        print("-"*100)
        #pdb.set_trace()

    # Create parent directory if it doesn't exist
    print(interaction_data_filename)
    os.makedirs(os.path.dirname(interaction_data_filename), exist_ok=True)
    json.dump(output_data, open(interaction_data_filename, 'w'), indent=2)

    #print(f"\n\nProlific ID{' '*25}\tBalance\tFalseAccepts\tFalseRejects\t# exits{' '*5}\t# of max bets\tAvg bet value")
    os.makedirs(os.path.dirname(batch_summaries_filename), exist_ok=True)
    f = open(batch_summaries_filename, 'w')
    f.write(f"Prolific ID\tSession ID\tBalance\tPaidBalance\t# exits\tStatus\n")
    for o in output_data:
        s = output_data[o]['interaction_summary']
        if output_data[o]['status'] in ['APPROVED', 'REJECTED']:
            f.write(f"{o}\t{output_data[o]['id']}\t${s['balance']:.2f}\t${max(s['balance'], 0):.2f}\t{s['num_exits']}\t{output_data[o]['status']}\n")
    f.close()

    # Read the tsv file into pandas dataframe and print 
    df = pd.read_csv(batch_summaries_filename.replace('.json', '.tsv'), sep='\t', header=0)
    print(df)