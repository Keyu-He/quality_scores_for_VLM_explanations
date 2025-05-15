# ece_analysis.py
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocessing
import time

def plot_calibration_curve(y_true, y_prob, ax, title):
    n_bins = 10
    bin_num_positives = []
    bin_mean_probs = []
    bin_num_instances = []
    bin_centers = []

    for i in range(n_bins):
        lower = i / n_bins
        upper = (i + 1) / n_bins if i != n_bins - 1 else 1.01
        idx = np.where((y_prob >= lower) & (y_prob < upper))[0]
        if len(idx) > 0:
            bin_num_positives.append(np.mean(y_true[idx]))
            bin_mean_probs.append(np.mean(y_prob[idx]))
            bin_num_instances.append(len(idx))
        else:
            bin_num_positives.append(0)
            bin_mean_probs.append(0)
            bin_num_instances.append(0)
        bin_centers.append(i + 0.5)

    total_instances = np.sum(bin_num_instances)
    ece = sum(count * abs(pos - prob)
              for pos, prob, count in zip(bin_num_positives, bin_mean_probs, bin_num_instances))
    ece = ece / total_instances if total_instances > 0 else None

    df = pd.DataFrame({
        'bin_centers': bin_centers,
        'accuracy': bin_num_positives,
        'counts': bin_num_instances
    })
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    cmap = plt.get_cmap('crest')
    sns.barplot(x='bin_centers', y='accuracy', data=df, ax=ax,
                hue='counts', palette='crest', hue_norm=(0,100),
                edgecolor='black', linewidth=1.2, width=1)
    ax.plot([-0.5, n_bins-0.5], [0, 1], 'k--')
    ax.set_xlim(-0.5, n_bins-0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Metric Score')
    ax.set_ylabel('Prediction Accuracy')
    ax.set_xticks(np.arange(n_bins))
    ax.set_xticklabels([f"{i/n_bins:.1f}" for i in range(n_bins)], rotation=45)
    ax.set_title(f"{title}\nECE = {ece:.4f}" if ece is not None else title)
    ax.legend().remove()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02).set_label('Number of Instances')
    return ece


def compute_ece(y_true, y_prob, n_bins=10):
    bin_num_positives = []
    bin_mean_probs = []
    bin_num_instances = []
    for i in range(n_bins):
        lower = i / n_bins
        upper = (i + 1) / n_bins if i != n_bins - 1 else 1.01
        idx = np.where((y_prob >= lower) & (y_prob < upper))[0]
        if len(idx) > 0:
            bin_num_positives.append(np.mean(y_true[idx]))
            bin_mean_probs.append(np.mean(y_prob[idx]))
            bin_num_instances.append(len(idx))
        else:
            bin_num_positives.append(0)
            bin_mean_probs.append(0)
            bin_num_instances.append(0)
    total = np.sum(bin_num_instances)
    ece = sum(count * abs(pos - prob)
              for pos, prob, count in zip(bin_num_positives, bin_mean_probs, bin_num_instances))
    return ece / total if total > 0 else None


def process_sample(args):
    i, pos_df, neg_df, output_base, sample_size = args
    per_class = sample_size // 2
    sample_dir = os.path.join(output_base, f'sample_{i}')
    os.makedirs(sample_dir, exist_ok=True)

    pos_samp = pos_df.sample(per_class, random_state=i)
    neg_samp = neg_df.sample(per_class, random_state=i)
    sample_df = pd.concat([pos_samp, neg_samp]).reset_index(drop=True)

    y_true = sample_df['is_correct'].astype(float).values
    vf = sample_df['visual_fidelity'].astype(float).values
    cs = sample_df['contrastiveness_score'].astype(float).values

    metrics = {
        'VisualFidelity': vf,
        'Contrastiveness': cs,
        'VFxContrastiveness': vf * cs,
        'Min(VF,Contrast)': np.minimum(vf, cs),
        'Avg(VF,Contrast)': (vf + cs) / 2,
        'Support': sample_df['entail_prob'].astype(float).values if 'entail_prob' in sample_df else None,
        'Informative': sample_df['informative'].astype(float).values if 'informative' in sample_df else None,
        'Commonsense_Plausibility': sample_df['commonsense_plausibility'].astype(float).values if 'commonsense_plausibility' in sample_df else None
    }

    rec = {'sample': i}
    for name, vals in metrics.items():
        if vals is None:
            continue
        ece_val = compute_ece(y_true, vals)
        rec[name] = ece_val

        # Calibration plot
        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        plot_calibration_curve(y_true, vals, ax, name)
        fig.tight_layout()
        fig.savefig(os.path.join(sample_dir, f"{name.replace('*','x')}.png"))
        plt.close(fig)
        
        # define 20 equal-width bins over [0,1]
        bin_edges = np.linspace(0, 1, 21)
        
        # Distributions
        sns.histplot(vals, bins=bin_edges, kde=True)
        plt.title(f"Distribution of {name} scores")
        plt.xlabel(name)
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(sample_dir, f"{name}_distribution.png"))
        plt.close()
        
        plt.figure(figsize=(6, 4), dpi=200)
        plt.title(f"Distribution of {name} scores for correct and incorrect answers")
        sns.histplot(vals[y_true == 1], bins=bin_edges, kde=False, label='Correct', color='tab:green')
        sns.histplot(vals[y_true == 0], bins=bin_edges, kde=False, label='Incorrect', color='tab:red')
        plt.xlabel(name)
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(os.path.join(sample_dir, f"{name}_correct_incorrect_distribution.png"))
        plt.close()

    return rec

if __name__ == '__main__':
    time_start = time.time()
    # model = 'qwen2.5-vl-7b-instruct'
    model = 'llava-v1.5-7b'
    dataset = 'AOKVQA'
    csv_path = os.path.join('model_outputs', dataset, f'{model}.csv')
    out_base = os.path.join('ece_analysis', f'{dataset}_random_sampled', model)
    n_samples = 50
    sample_size = 100

    # Load data once
    data = pd.read_csv(csv_path)
    pos_df = data[data.is_correct == 1]
    neg_df = data[data.is_correct == 0]

    # Prepare arguments for parallel processing
    args_list = [(i, pos_df, neg_df, out_base, sample_size) for i in range(1, n_samples + 1)]

    with multiprocessing.Pool(processes=n_samples) as pool:
        records = pool.map(process_sample, args_list)

    # Save summary and identify best run
    summary_df = pd.DataFrame(records)
    
    # print ECEs
    for row in summary_df.iterrows():
        print(f"Run {int(row[1]['sample']):02}: ECE VF={row[1]['VisualFidelity']:.4f}, CR={row[1]['Contrastiveness']:.4f}, Avg={row[1]['Avg(VF,Contrast)']:.4f}", end=', ')
        print(f"Min={row[1]['Min(VF,Contrast)']:.4f}, PROD={row[1]['VFxContrastiveness']:.4f}", end=', ')
        print(f"SPT={row[1]['Support']:.4f}, INFO={row[1]['Informative']:.4f}, COMSP={row[1]['Commonsense_Plausibility']:.4f}")

    cols = ['VisualFidelity', 'Contrastiveness', 'Avg(VF,Contrast)']
    summary_df['mean_ece'] = summary_df[cols].mean(axis=1)
    best = summary_df.loc[summary_df['mean_ece'].idxmin()]
    print(f"Best run: sample_{int(best['sample'])} with mean ECE among VF, CONTR, and AVG={best['mean_ece']:.4f}")

    os.makedirs(out_base, exist_ok=True)
    summary_df.to_csv(os.path.join(out_base, 'ece_summary.csv'), index=False)
    print(f"Completed {n_samples} runs. Summary saved to {out_base}/ece_summary.csv")
    print(f"Total time taken: {time.time() - time_start:.2f} seconds")
