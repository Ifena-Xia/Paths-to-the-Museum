import os, json
import numpy as np
import matplotlib.pyplot as plt

OUT = "figs"; os.makedirs(OUT, exist_ok=True)
PREFIX = "paths_museum"
CEPS_JSON = "ceps_final_translated.json"   # put the analytic file beside this script for figure C

# ---- palette: RED accent, gray baseline ----
INK="#121212"; MUTED="#6b6b6b"; GRID="#dcdcdc"
ACCENT="#C9252D"      # red, the "this is the point" color for Paths
ACCENT2="#F1A9A0"     # light red
NEUTRAL="#8a8f94"     # gray baseline
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.edgecolor":INK,
    "axes.linewidth":0.8,"axes.grid":True,"grid.color":GRID,"grid.linewidth":0.7,
    "axes.axisbelow":True,"svg.fonttype":"none"})

def _frame(fig,title,subtitle,source):
    fig.text(0.065,0.965,title,ha="left",va="top",fontsize=15,fontweight="bold",color=INK)
    fig.text(0.065,0.915,subtitle,ha="left",va="top",fontsize=10.3,color=INK)
    fig.text(0.065,0.018,source,ha="left",va="bottom",fontsize=8.5,color=MUTED)
    fig.add_artist(plt.Line2D([0.065,0.115],[0.986,0.986],color=ACCENT,linewidth=3.2,solid_capstyle="butt"))

def figA():
    b=dict(c=2.006,edu=0.0401,semi=-0.4773,rural=-0.9231,edu_semi=0.0293,edu_rural=0.0391,
           female=0.0561,grade9=-0.1813,sib=-0.1251,std=0.0655)
    edu=np.linspace(6,19,100); female,sib,std=0,0,1
    def pred(e,reg,g9):
        y=b["c"]+b["edu"]*e+b["female"]*female+b["grade9"]*(1 if g9 else 0)+b["sib"]*sib+b["std"]*std
        if reg=="semi": y+=b["semi"]+b["edu_semi"]*e
        elif reg=="rural": y+=b["rural"]+b["edu_rural"]*e
        return y
    panels=[("Urban core",None),("Semi-urban","semi"),("Rural","rural")]
    fig,axes=plt.subplots(1,3,figsize=(10.5,4.2),sharey=True)
    fig.subplots_adjust(left=0.065,right=0.975,top=0.80,bottom=0.22,wspace=0.12)
    for ax,(lab,reg) in zip(axes,panels):
        y7,y9=pred(edu,reg,False),pred(edu,reg,True)
        ax.plot(edu,y7,color=ACCENT,linewidth=2.4)
        ax.plot(edu,y9,color=NEUTRAL,linewidth=2.2,linestyle=(0,(4,2)))
        ax.set_title(lab,fontsize=11,fontweight="bold",loc="left",pad=6)
        ax.set_xlim(6,19); ax.set_xlabel("Parental education, years",fontsize=9.5)
        ax.spines[["top","right"]].set_visible(False)
        if reg=="rural":
            ax.text(19.1,y7[-1],"Grade 7",color=ACCENT,fontsize=9,va="center",fontweight="bold")
            ax.text(19.1,y9[-1],"Grade 9",color=NEUTRAL,fontsize=9,va="center",fontweight="bold")
    axes[0].set_ylabel("Predicted frequency (1\u20136)",fontsize=9.5)
    _frame(fig,"Education narrows the rural gap",
        "Predicted cultural-participation frequency by parental education, OLS Model 2. Held fixed: male, no siblings, both parents.",
        "Source: author's estimates from CEPS 2013\u201314 (coefficients verified against full re-estimation)")
    for e in ("svg","png"): fig.savefig(f"{OUT}/{PREFIX}_figA_predicted_frequency.{e}",dpi=200,bbox_inches="tight")
    plt.close(fig)

def figB():
    z1=dict(c=-1.9683,mid=0.4472,high=0.8168,female=0.1981,g9=-0.2983,sib=-0.4259,std=0.2592,semi=0.5915,urb=0.6607)
    z2=dict(c=-2.466,mid=0.2178,high=0.3668,female=0.042,g9=-0.4236,sib=-0.1055,std=0.302,semi=0.5846,urb=0.9812)
    female,g9,sib,std,semi,urb=0,0,0,1,0,1
    def lin(b,m,h): return (b["c"]+b["mid"]*m+b["high"]*h+b["female"]*female+b["g9"]*g9+b["sib"]*sib+b["std"]*std+b["semi"]*semi+b["urb"]*urb)
    bands=[("Low\n(\u22649 yrs)",0,0),("Medium\n(~12 yrs)",1,0),("High\n(\u226515 yrs)",0,1)]
    P=[]
    for _,m,h in bands:
        Z1,Z2=lin(z1,m,h),lin(z2,m,h); den=1+np.exp(Z1)+np.exp(Z2)
        P.append((1/den,np.exp(Z1)/den,np.exp(Z2)/den))
    P=np.array(P)
    fig,ax=plt.subplots(figsize=(7.6,4.6)); fig.subplots_adjust(left=0.10,right=0.80,top=0.80,bottom=0.24)
    x=np.arange(3); cats=["Low participation","Moderate","High participation"]; cols=[NEUTRAL,ACCENT2,ACCENT]
    bottom=np.zeros(3)
    for j,(cat,col) in enumerate(zip(cats,cols)):
        v=P[:,j]*100; ax.bar(x,v,bottom=bottom,color=col,width=0.62,edgecolor="white",linewidth=1.2)
        for i in range(3):
            if v[i]>5: ax.text(x[i],bottom[i]+v[i]/2,f"{v[i]:.0f}",ha="center",va="center",color="white",fontsize=9.5,fontweight="bold")
        ax.text(2.42,bottom[-1]+v[-1]/2,cat,color=col,fontsize=9.5,va="center",fontweight="bold"); bottom+=v
    ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bands],fontsize=10); ax.set_ylim(0,100)
    ax.set_ylabel("Predicted probability, %",fontsize=9.5); ax.set_xlabel("Parental-education band",fontsize=9.5)
    ax.spines[["top","right"]].set_visible(False)
    _frame(fig,"As parents' schooling rises, the mix shifts up",
        "Predicted probability of low, moderate, and high participation by parental-education band, multinomial logit.",
        "Source: author's estimates from CEPS 2013\u201314")
    for e in ("svg","png"): fig.savefig(f"{OUT}/{PREFIX}_figB_multinomial_mix.{e}",dpi=200,bbox_inches="tight")
    plt.close(fig)

def figC():
    if not os.path.exists(CEPS_JSON):
        print("  (figC skipped: place", CEPS_JSON, "beside the script)"); return
    import pandas as pd
    df=pd.DataFrame(json.load(open(CEPS_JSON)))
    df=df[df['living_area']!='Other'].copy()
    fmap={'Never':1,'Once a year':2,'Once every half year':3,'Once a month':4,'Once a week':5,'More than once a week':6}
    em={'None':0,'Finished elementary school':6,'Junior high school':9,'Technical secondary school or technical school':12,
        'Vocational high school':12,'Senior high school':12,'Junior college':15,'Bachelor degree':16,'Master degree or higher':19}
    um={'Central area of the city/county':'Urban core','Outskirts of the city/county':'Semi-urban',
        'The \u201crural-urban continuum\u201d area of the city/county':'Semi-urban','Towns outside the city/county':'Semi-urban','Rural area':'Rural'}
    df['freq']=df['frequency'].map(fmap)
    df['edu']=df[['mo_edu','fa_edu']].apply(lambda r:max(em[r['mo_edu']],em[r['fa_edu']]),axis=1)
    df['urb']=df['living_area'].map(um)
    df['both_parents']=((df['pr_w_m']==1)&(df['pr_w_f']==1)).astype(int)

    fig,axes=plt.subplots(2,2,figsize=(9.6,7.2))
    fig.subplots_adjust(left=0.08,right=0.96,top=0.84,bottom=0.08,hspace=0.5,wspace=0.28)
    flabels=["Never","Once a\nyear","Half-\nyearly","Monthly","Weekly","More"]
    fc=df['freq'].value_counts().reindex(range(1,7)).values
    axes[0,0].bar(range(1,7),fc,color=ACCENT,width=0.8); axes[0,0].set_xticks(range(1,7)); axes[0,0].set_xticklabels(flabels,fontsize=8)
    axes[0,0].set_title("Participation frequency",loc="left",fontweight="bold",fontsize=11.5)
    axes[0,1].hist(df['edu'],bins=[ -0.5,3,7.5,10.5,13.5,16.5,20],color=NEUTRAL,rwidth=0.92)
    axes[0,1].set_title("Parental education, years",loc="left",fontweight="bold",fontsize=11.5)
    uo=["Urban core","Semi-urban","Rural"]; uc=df['urb'].value_counts().reindex(uo).values
    axes[1,0].bar(uo,uc,color=[ACCENT,ACCENT2,NEUTRAL]); axes[1,0].set_title("Urbanicity",loc="left",fontweight="bold",fontsize=11.5)
    for t in axes[1,0].get_xticklabels(): t.set_fontsize(9)
    lv=df['both_parents'].value_counts().reindex([1,0]).values
    axes[1,1].bar(["Both parents","Other"],lv,color=[ACCENT,NEUTRAL]); axes[1,1].set_title("Lives with both parents",loc="left",fontweight="bold",fontsize=11.5)
    for ax in axes.ravel(): ax.spines[["top","right"]].set_visible(False)
    _frame(fig,"Who is in the sample",
        f"CEPS analytic sample, {len(df):,} families, distributions of key variables.",
        "Source: author's calculations from CEPS 2013\u201314")
    for e in ("svg","png"): fig.savefig(f"{OUT}/{PREFIX}_figC_descriptives.{e}",dpi=200,bbox_inches="tight")
    plt.close(fig); print("  figC written (N=%d)"%len(df))

if __name__=="__main__":
    figA(); figB(); figC()
    print("Wrote figures with prefix", PREFIX, "to", os.path.abspath(OUT))
