import numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
mpl.rcParams.update({"font.family":"serif","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":200})
NAVY="#1F3A5F"; ORANGE="#D9822B"; GREY="#8A8F98"; TEAL="#2A9D8F"; RED="#B5443A"

# ---- Fig: DTW illustration (synthetic) ----
from numpy.random import default_rng
rng=default_rng(3)
t=np.arange(31)
base=np.cumsum(np.r_[0, -3.0, -1.5, rng.normal(-0.2,0.5,28)])
q=(base-base.mean())/base.std()
# candidate: same shape but the jump is spread over 3 min and later drift slower
tt=np.linspace(0,1,31); warp=tt**1.25
c=np.interp(warp*30, t, base)+rng.normal(0,0.25,31)
c=(c-c.mean())/c.std()
def dtw(a,b,band=5):
    n=len(a);D=np.full((n+1,n+1),np.inf);D[0,0]=0
    for i in range(1,n+1):
        for j in range(max(1,i-band),min(n,i+band)+1):
            D[i,j]=(a[i-1]-b[j-1])**2+min(D[i-1,j],D[i,j-1],D[i-1,j-1])
    i,j=n,n;path=[(i-1,j-1)]
    while (i,j)!=(1,1):
        opts=[(D[i-1,j-1],i-1,j-1),(D[i-1,j],i-1,j),(D[i,j-1],i,j-1)]
        _,i,j=min(opts);path.append((i-1,j-1))
    return np.sqrt(D[n,n]),path[::-1]
d,path=dtw(q,c)
fig,ax=plt.subplots(1,2,figsize=(10,3.4))
off=3.2
ax[0].plot(t,q+off,color=NAVY,lw=2,label="Today's path")
ax[0].plot(t,c,color=ORANGE,lw=2,label="Past path")
for i,j in path[::2]:
    ax[0].plot([i,j],[q[i]+off,c[j]],color=GREY,lw=0.6,alpha=0.7)
ax[0].set_yticks([]);ax[0].set_xlabel("Minute in the 31-minute path");ax[0].legend(frameon=False,loc="upper right",fontsize=8)
ax[0].set_title("(a) DTW aligns points of similar shape",fontsize=10)
n=31;D=np.array([[ (q[i]-c[j])**2 for j in range(n)] for i in range(n)])
mask=np.array([[abs(i-j)>5 for j in range(n)] for i in range(n)])
Dm=np.ma.array(D,mask=mask)
ax[1].imshow(Dm,origin="lower",cmap="Blues",aspect="auto")
pi,pj=zip(*path);ax[1].plot(pj,pi,color=ORANGE,lw=2)
ax[1].plot([0,30],[0,30],color=GREY,lw=0.8,ls="--")
ax[1].set_xlabel("Past path, minute");ax[1].set_ylabel("Today's path, minute")
ax[1].set_title("(b) Warping path inside a band of $\\pm5$ minutes",fontsize=10)
ax[1].spines[["top","right"]].set_visible(True)
plt.tight_layout();plt.savefig("figures/dtw_illustration.png");plt.close()

# ---- Fig: event timing / ladder ----
fig,ax=plt.subplots(figsize=(10,2.6))
ax.axhline(0,color="black",lw=0.8)
ax.axvspan(-30,1,ymin=0.55,ymax=0.85,color=NAVY,alpha=0.25)
ax.text(-14.5,0.7,"Path, window 1\n$[T-30,\\,T+1]$",ha="center",va="center",fontsize=9,transform=ax.get_xaxis_transform())
cols=[TEAL,"#5BB5A9","#8CCBC1","#B7DDD6","#D8ECE8"]
for k,a in enumerate([1,31,61,91,121]):
    ax.axvspan(a,a+30,ymin=0.15,ymax=0.45,color=cols[k],alpha=0.9)
    ax.text(a+15,0.3,f"window {k+1}\n$T{{+}}{a}\\to T{{+}}{a+30}$",ha="center",va="center",fontsize=8,transform=ax.get_xaxis_transform())
ax.axvline(0,color=RED,lw=1.5);ax.text(0,0.95,"release $T$",color=RED,ha="center",fontsize=9,transform=ax.get_xaxis_transform())
ax.axvline(1,color=NAVY,lw=1,ls=":")
ax.set_xlim(-35,155);ax.set_yticks([]);ax.set_xlabel("Minutes from the release")
ax.spines[["left"]].set_visible(False)
plt.tight_layout();plt.savefig("figures/event_ladder.png");plt.close()

# ---- Fig: ICs train vs valid ----
fig,ax=plt.subplots(1,2,figsize=(10,3.2),sharey=True)
lab=["ES","NQ"];x=np.arange(2);w=0.35
tr=[-0.060,-0.042];va=[0.069,0.091]
ax[0].bar(x-w/2,tr,w,color=GREY,label="Train 2013–2017");ax[0].bar(x+w/2,va,w,color=NAVY,label="Validation 2018–2020")
ax[0].axhline(0,color="black",lw=0.8);ax[0].set_xticks(x,lab);ax[0].set_title("(a) Direction: rank IC of $\\mu$ with the next return",fontsize=10)
ax[0].legend(frameon=False,fontsize=8,loc="upper left")
tr=[0.137,0.242];va=[0.162,0.102]
ax[1].bar(x-w/2,tr,w,color=GREY);ax[1].bar(x+w/2,va,w,color=NAVY)
ax[1].axhline(0,color="black",lw=0.8);ax[1].set_xticks(x,lab);ax[1].set_title("(b) Size: rank IC of $\\sigma$ with the absolute next return",fontsize=10)
for a in ax:
    for p in a.patches:
        h=p.get_height();a.text(p.get_x()+p.get_width()/2,h+(0.006 if h>=0 else -0.02),f"{h:+.3f}",ha="center",fontsize=8)
plt.tight_layout();plt.savefig("figures/dtw_ic.png");plt.close()

# ---- Fig: Sharpe by cost ----
fig,ax=plt.subplots(figsize=(8,3.4))
books=["ES","NQ","ES + NQ"];x=np.arange(3);w=0.19
vals={"Gross":[0.78,0.14,0.64],"Quarter tick":[0.64,0.11,0.52],"Half tick":[0.49,0.08,0.40],"Buy-and-hold":[0.69,0.58,0.64]}
cc=[NAVY,"#4F6D91","#8FA6C1",ORANGE]
for i,(k,v) in enumerate(vals.items()):
    b=ax.bar(x+(i-1.5)*w,v,w,color=cc[i],label=k)
    for r in b: ax.text(r.get_x()+r.get_width()/2,r.get_height()+0.01,f"{r.get_height():.2f}",ha="center",fontsize=7.5)
ax.set_xticks(x,books);ax.set_ylabel("Sharpe ratio, 2021–2026");ax.legend(frameon=False,fontsize=8,ncol=4,loc="upper right")
ax.set_ylim(0,0.95)
plt.tight_layout();plt.savefig("figures/sharpe_by_cost.png");plt.close()

# ---- Fig: robustness heatmap ----
M=np.array([[0.36,0.61,0.42],[0.50,0.50,0.51],[0.17,0.64,0.44],[0.34,0.61,0.36]])
rows=["Outer 60%, cut-offs fixed","Outer 60%, cut-offs refit yearly","Outer 80%, cut-offs fixed","Outer 80%, cut-offs refit yearly"]
fig,ax=plt.subplots(figsize=(7,3.0))
im=ax.imshow(M,cmap="Blues",vmin=0,vmax=0.8,aspect="auto")
ax.set_xticks(range(3),["$k=10$","$k=15$","$k=25$"]);ax.set_yticks(range(4),rows)
for i in range(4):
    for j in range(3): ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",color="white" if M[i,j]>0.45 else "black",fontsize=9)
ax.spines[:].set_visible(False)
cb=plt.colorbar(im,ax=ax,fraction=0.04);cb.set_label("ES Sharpe, quarter tick",fontsize=8)
plt.tight_layout();plt.savefig("figures/robustness_es.png");plt.close()
print("ok",d)
