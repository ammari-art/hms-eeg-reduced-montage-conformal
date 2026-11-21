import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
# Table 2 (mean, sd) transcribed from the Kaggle log, version 2 (scriptVersionId 372118120)
auc = {"full18_kl":(0.920,0.014),"temporal8_kl":(0.902,0.021),"parasagittal8_kl":(0.893,0.012),"frontotemporal4_kl":(0.873,0.050),
       "frontal2_kl":(0.846,0.041),"learned_k2_kl":(0.848,0.016),"learned_k4_kl":(0.857,0.008),"learned_k6_kl":(0.879,0.025),
       "learned_k8_kl":(0.895,0.017),"learned_k10_kl":(0.903,0.022)}
kl = {"full18_kl":(0.586,0.047),"temporal8_kl":(0.660,0.047),"parasagittal8_kl":(0.674,0.060),"frontotemporal4_kl":(0.745,0.126),
      "frontal2_kl":(0.812,0.078),"learned_k2_kl":(0.812,0.077),"learned_k4_kl":(0.795,0.034),"learned_k6_kl":(0.722,0.084),
      "learned_k8_kl":(0.674,0.039),"learned_k10_kl":(0.642,0.075)}
k = {"full18_kl":18,"temporal8_kl":8,"parasagittal8_kl":8,"frontotemporal4_kl":4,"frontal2_kl":2,
     "learned_k2_kl":2,"learned_k4_kl":4,"learned_k6_kl":6,"learned_k8_kl":8,"learned_k10_kl":10}
for n in auc: print(f"{n:20s} AUROC retained {100*auc[n][0]/0.920:5.1f}%  KL +{kl[n][0]-0.586:.3f}")
sel = {"k2":{0:['C3-P3','C4-P4'],1:['T3-T5','T3-T5'],2:['F3-C3','F3-C3'],3:['T3-T5','T4-T6'],4:['T3-T5','F4-C4']},
 "k4":{0:['C3-P3','C3-P3','C4-P4','C3-P3'],1:['T3-T5','T4-T6','T4-T6','T4-T6'],2:['T4-T6','F4-C4','T4-T6','T4-T6'],3:['T3-T5','T3-T5','T4-T6','T4-T6'],4:['F7-T3','F4-C4','F3-C3','F4-C4']},
 "k6":{0:['C3-P3','T3-T5','T3-T5','C3-P3','C4-P4','C3-P3'],1:['T4-T6','T3-T5','T3-T5','C4-P4','T3-T5','T3-T5'],2:['F4-C4','T4-T6','T3-T5','T3-T5','F3-C3','T4-T6'],3:['T3-T5','T4-T6','T4-T6','F3-C3','T4-T6','T4-T6'],4:['F4-C4','T4-T6','F3-C3','F4-C4','F3-C3','F3-C3']},
 "k8":{0:['C3-P3','F4-C4','T4-T6','F3-C3','C3-P3','C3-P3','C4-P4','T3-T5'],1:['C4-P4','C3-P3','C3-P3','F8-T4','F7-T3','C3-P3','T3-T5','T4-T6'],2:['T3-T5','T4-T6','T4-T6','C4-P4','T4-T6','T3-T5','T3-T5','C3-P3'],3:['C4-P4','C3-P3','F3-C3','F4-C4','F4-C4','F3-C3','T3-T5','T3-T5'],4:['F8-T4','T4-T6','T3-T5','F3-C3','F3-C3','F4-C4','T4-T6','T3-T5']},
 "k10":{0:['F4-C4','F4-C4','T4-T6','F3-C3','T3-T5','T4-T6','F3-C3','C3-P3','F8-T4','C3-P3'],1:['T4-T6','F3-C3','C4-P4','F7-T3','T3-T5','C3-P3','F3-C3','C4-P4','T4-T6','F4-C4'],2:['C3-P3','T3-T5','Fz-Cz','C4-P4','F4-C4','C3-P3','F7-T3','F4-C4','C3-P3','T4-T6'],3:['F7-T3','T4-T6','T3-T5','C4-P4','F3-C3','C3-P3','C4-P4','T3-T5','F3-C3','F8-T4'],4:['F3-C3','T4-T6','F3-C3','F4-C4','C4-P4','F8-T4','Fz-Cz','F8-T4','F7-T3','C4-P4']}}
from collections import Counter
rows = []
for kk, d in sel.items():
    uniq = [len(set(v)) for v in d.values()]
    cnt = Counter(c for v in d.values() for c in set(v))
    rows.append((kk, uniq, cnt.most_common(5)))
    print(kk, "unique per fold", uniq, "mean", np.mean(uniq), "| folds containing:", cnt.most_common(6))
json.dump({"sel_unique": {r[0]: r[1] for r in rows}}, open("sel.json","w"))
# Figure 2
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
L = ["learned_k2_kl","learned_k4_kl","learned_k6_kl","learned_k8_kl","learned_k10_kl","full18_kl"]
Fx = ["frontal2_kl","frontotemporal4_kl","temporal8_kl","parasagittal8_kl"]
names = {"frontal2_kl":"Frontal","frontotemporal4_kl":"Fronto-temporal","temporal8_kl":"Temporal chains","parasagittal8_kl":"Parasagittal","full18_kl":"Full"}
for a, d, lab in [(ax[0], auc, "Macro-AUROC"), (ax[1], kl, "KL divergence (lower is better)")]:
    xu = {"learned_k2_kl":1.6,"learned_k4_kl":2.2,"learned_k6_kl":3.2,"learned_k8_kl":5.2,"learned_k10_kl":6.8,"full18_kl":18}
    a.errorbar([xu[n] for n in L], [d[n][0] for n in L], yerr=[d[n][1] for n in L], fmt="o-", color="#2a6fb0", capsize=3, label="Learned subset (plotted at mean distinct channels)")
    a.errorbar([k[n]+0.25 for n in Fx], [d[n][0] for n in Fx], yerr=[d[n][1] for n in Fx], fmt="s", color="#d1603d", capsize=3, label="Fixed clinical montage")
    for n in Fx: a.annotate(names[n], (k[n]+0.25, d[n][0]), textcoords="offset points", xytext=(6, -3 if lab.startswith("KL") else 3), fontsize=8, color="#d1603d")
    a.set_xlabel("Distinct bipolar channels used"); a.set_ylabel(lab); a.set_xticks([2,4,6,8,10,18]); a.grid(alpha=.3)
ax[0].legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig("fig2_channels_curve.png", dpi=300)
# Figure 4 occlusion
ch = ["Fp1-F7","F7-T3","T3-T5","T5-O1","Fp2-F8","F8-T4","T4-T6","T6-O2","Fp1-F3","F3-C3","C3-P3","P3-O1","Fp2-F4","F4-C4","C4-P4","P4-O2","Fz-Cz","Cz-Pz"]
occ = np.array([[0.0113,-0.0003,0.0446,-0.0039,0.0288,-0.0063],[-0.0098,0.0113,0.0196,0.0006,0.0231,-0.0042],[0.0000,0.0117,0.0249,0.0054,0.0249,-0.0077],
 [0.0283,0.0186,-0.0062,-0.0115,0.0029,-0.0004],[0.0073,0.0300,0.0384,-0.0168,0.0219,-0.0082],[-0.0060,-0.0028,0.0225,-0.0183,0.0294,-0.0022],
 [-0.0011,-0.0009,0.0297,-0.0033,0.0379,-0.0094],[0.0233,0.0099,-0.0074,-0.0091,0.0044,0.0004],[0.0356,-0.0064,0.0442,-0.0271,0.0328,-0.0073],
 [0.0090,-0.0124,0.0794,-0.0161,0.0453,-0.0104],[-0.0006,0.0138,-0.0029,-0.0033,0.0076,-0.0026],[0.0296,0.0173,0.0015,-0.0106,0.0056,-0.0001],
 [0.0364,-0.0133,0.0555,-0.0172,0.0309,-0.0081],[0.0099,-0.0109,0.0932,-0.0191,0.0521,-0.0119],[-0.0035,0.0191,-0.0017,0.0014,-0.0018,-0.0029],
 [0.0279,0.0038,-0.0046,-0.0152,0.0066,-0.0000],[0.0451,-0.0309,0.0613,-0.0192,0.0494,-0.0029],[0.0419,-0.0081,0.0003,-0.0172,0.0116,0.0002]])
fig, a = plt.subplots(figsize=(5.6, 6.4)); v = np.abs(occ).max()
im = a.imshow(occ, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
a.set_xticks(range(6), ["Seizure","LPD","GPD","LRDA","GRDA","Other"]); a.set_yticks(range(18), ch)
for i in range(18):
    for j in range(6): a.text(j, i, f"{occ[i,j]:.3f}", ha="center", va="center", fontsize=6.5, color="white" if abs(occ[i,j]) > 0.6*v else "black")
fig.colorbar(im, label="Drop in p(majority class) when channel is zeroed"); a.set_title("Channel occlusion, full 18-channel KL model", fontsize=10)
fig.tight_layout(); fig.savefig("fig4_occlusion.png", dpi=300)
print("figs ok")
