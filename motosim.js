/* MotoSim — ideal line + minimum-lap-time optimisation for a motorcycle.
   Pure functions, no DOM. Units: metres, seconds, kg, W, radians unless noted.

   Objective (time integral in curvilinear coordinates along the centreline s):
       T = ∫ sqrt( (1 − n(s)·κc(s))² + (dn/ds)² ) / v(s) ds
   Constraints:
     • corridor      lo(s) ≤ n(s) ≤ hi(s)   (road edges minus half the bike width, or own lane)
     • lean / grip   v²·κ ≤ min(μ, tan θmax) · g_eff,   θ = arctan(v²κ / g_eff)
     • friction circle  (a_long/μg_eff)² + (a_lat/μg_eff)² ≤ 1   (on tyre forces)
     • power         F_drive ≤ P·η / v
     • wheelie / stoppie  a ≤ g_eff·b/h ,  braking ≤ g_eff·a/h
   g_eff = g·cosφ + v²·κv  (grade φ, vertical curvature κv: crests unload the tyres). */
var MotoSim = (function(){
"use strict";
var G = 9.81;

function gaussKernel(sig){
  if(!(sig>0.3)) return {R:0, k:new Float64Array([1])};
  var R=Math.ceil(sig*4), k=new Float64Array(2*R+1), s=0, j;
  for(j=-R;j<=R;j++){ var v=Math.exp(-0.5*(j/sig)*(j/sig)); k[j+R]=v; s+=v; }
  for(j=0;j<k.length;j++) k[j]/=s;
  return {R:R, k:k};
}
/* 1-D Gaussian filter, edge mode 'nearest' (as scipy.ndimage.gaussian_filter1d mode='nearest'). sig in samples */
function gauss(a, sig){
  var n=a.length, out=new Float64Array(n), K=gaussKernel(sig), R=K.R, k=K.k;
  for(var i=0;i<n;i++){ var acc=0; for(var j=-R;j<=R;j++){ var t=i+j; if(t<0) t=0; else if(t>=n) t=n-1; acc+=a[t]*k[j+R]; } out[i]=acc; }
  return out;
}
function deriv(a, ds){
  var n=a.length, d1=new Float64Array(n), d2=new Float64Array(n), i;
  for(i=0;i<n;i++){ var lo=Math.max(0,i-1), hi=Math.min(n-1,i+1); d1[i]=(a[hi]-a[lo])/((hi-lo)*ds); }
  for(i=1;i<n-1;i++) d2[i]=(a[i+1]-2*a[i]+a[i-1])/(ds*ds);
  if(n>2){ d2[0]=d2[1]; d2[n-1]=d2[n-2]; }
  return [d1,d2];
}
/* κ = (x'y'' − y'x'') / (x'² + y'²)^(3/2), plus unit tangent */
function curvature(x, y, ds){
  var dx=deriv(x,ds), dy=deriv(y,ds), n=x.length, k=new Float64Array(n), tx=new Float64Array(n), ty=new Float64Array(n);
  for(var i=0;i<n;i++){
    var x1=dx[0][i], x2=dx[1][i], y1=dy[0][i], y2=dy[1][i], q=x1*x1+y1*y1;
    if(q<1e-12){ tx[i]=1; continue; }
    k[i]=(x1*y2-y1*x2)/Math.pow(q,1.5); var L=Math.sqrt(q); tx[i]=x1/L; ty[i]=y1/L;
  }
  return {k:k, tx:tx, ty:ty};
}
function circleCurv(ax,ay,bx,by,cx,cy){
  var cr=(bx-ax)*(cy-by)-(by-ay)*(cx-bx);
  var a=Math.hypot(bx-ax,by-ay), b=Math.hypot(cx-bx,cy-by), c=Math.hypot(cx-ax,cy-ay), den=a*b*c;
  return den>1e-9 ? 2*cr/den : 0;
}

/* geometry shared by every line on the same road */
function prepare(inp, p){
  var X=inp.X, Y=inp.Y, E=inp.E, ds=inp.ds, N=X.length, i;
  var Cx=gauss(X,p.sigmaCenter/ds), Cy=gauss(Y,p.sigmaCenter/ds);      // smooth polyline kinks
  var c0=curvature(Cx,Cy,ds);
  var Nx=new Float64Array(N), Ny=new Float64Array(N), dc=new Float64Array(N);
  for(i=0;i<N;i++){ Nx[i]=-c0.ty[i]; Ny[i]=c0.tx[i]; }                  // left-hand normal: n > 0 is left of travel
  for(i=0;i<N-1;i++) dc[i]=Math.hypot(Cx[i+1]-Cx[i],Cy[i+1]-Cy[i]);
  var Eg=gauss(E,15/ds), Ev=gauss(E,40/ds), dE=deriv(Eg,ds), dV=deriv(Ev,ds);
  var phi=new Float64Array(N), kv=new Float64Array(N);
  for(i=0;i<N;i++){ phi[i]=Math.atan(dE[0][i]); kv[i]=dV[1][i]/Math.pow(1+dV[0][i]*dV[0][i],1.5); }
  return {N:N, ds:ds, Cx:Cx, Cy:Cy, Nx:Nx, Ny:Ny, kc:c0.k, dc:dc, phi:phi, kv:kv};
}

/* vehicle model: per-point limits and the forward/backward integration steps (shared by evaluation and optimiser) */
function model(g, inp, p){
  var m=p.mass, Pw=p.powerW*p.eta, kD=0.5*p.rho*p.CdA, mu=p.mu, Crr=p.Crr, kv=g.kv;
  var latCoef=Math.min(mu, Math.tan(p.leanMax*Math.PI/180));
  var wheelie=p.wheelbase*p.frontFrac/p.cogH, stoppie=p.wheelbase*(1-p.frontFrac)/p.cogH;
  var cosP=new Float64Array(g.N), sinP=new Float64Array(g.N);
  for(var i=0;i<g.N;i++){ cosP[i]=Math.cos(g.phi[i]); sinP[i]=Math.sin(g.phi[i]); }
  var legal=inp.legal, useLegal=p.useLegal, vCap=p.vCap;
  var tanLean=Math.tan(p.leanMax*Math.PI/180);
  function limits(i,k){
    var gN=G*cosP[i], lim=Infinity, ty='none', ak=Math.abs(k);
    // cornering limit v ≤ √(μ g R_eff), with g_eff and with the share of the friction circle that is needed just to
    // hold the speed (gravity on a downhill, drag and rolling resistance) — solved by a short fixed-point iteration
    var c=latCoef, vl=Infinity;
    for(var it=0; it<5; it++){
      var den=ak-c*kv[i]; if(!(den>1e-9)){ vl=Infinity; break; }
      vl=Math.sqrt(c*gN/den);
      var gE=gN+vl*vl*kv[i]; if(gE<0.1) gE=0.1;
      var aReq=G*sinP[i]+kD*vl*vl/m+G*Crr*cosP[i], lat=(mu*gE)*(mu*gE)-aReq*aReq;
      var cn=Math.min(tanLean, lat>0?Math.sqrt(lat)/gE:0); if(Math.abs(cn-c)<1e-6){ c=cn; break; } c=cn;
    }
    if(vl<lim){ lim=vl; ty='corner'; }
    if(kv[i]<-1e-9){ var vc=Math.sqrt(gN/(-kv[i])); if(vc<lim){ lim=vc; ty='crest'; } }   // crest: tyre load stays ≥ 0
    if(useLegal && legal && legal[i]>0 && legal[i]<lim){ lim=legal[i]; ty='legal'; }
    if(vCap>0 && vCap<lim){ lim=vCap; ty='cap'; }
    return [lim,ty];
  }
  function gEf(i,v){ var gE=G*cosP[i]+v*v*kv[i]; return gE<0.1?0.1:gE; }
  function avail(i,v,k){ var gE=gEf(i,v), aLat=v*v*Math.abs(k), av=(mu*gE)*(mu*gE)-aLat*aLat; return av>0?Math.sqrt(av):0; }   // friction circle
  return {
    latCoef:latCoef,
    vmax:function(i,k){ return limits(i,k)[0]; },
    vtype:function(i,k){ return limits(i,k)[1]; },
    /* forward step i → i+1 (accelerating as hard as allowed). The friction circle is checked at BOTH ends of the
       segment (speed at i+1 is predicted first), so grip is not exceeded where curvature changes along the segment. */
    fwd:function(v, i, k, k1, dsT, vmaxNext){
      var aT=avail(i,v,k), F=Math.min(Pw/(v>1?v:1), m*aT, m*gEf(i,v)*wheelie);
      var acc=(F-kD*v*v-m*G*Crr*cosP[i])/m-G*sinP[i], v2=v*v+2*acc*dsT, vn=v2>0?Math.sqrt(v2):0;
      var aT1=avail(i+1,vn,k1);
      if(aT1<aT){ F=Math.min(Pw/(v>1?v:1), m*aT1, m*gEf(i,v)*wheelie); acc=(F-kD*v*v-m*G*Crr*cosP[i])/m-G*sinP[i]; v2=v*v+2*acc*dsT; vn=v2>0?Math.sqrt(v2):0; }
      return vn<vmaxNext?vn:vmaxNext;
    },
    /* backward step: highest speed at i that can still brake down to u at i+1 (stoppie and friction circle at both ends) */
    bwd:function(u, i, k, k1, dsT, vfi){
      var aB=avail(i+1,u,k1), Fb=Math.min(m*aB, m*gEf(i+1,u)*stoppie);
      var dec=Fb/m+kD*u*u/m+G*Crr*cosP[i]+G*sinP[i]; if(dec<0) dec=0;
      var vb=Math.sqrt(u*u+2*dec*dsT); if(vb>vfi) vb=vfi;
      var aB0=avail(i,vb,k);
      if(aB0<aB){ Fb=Math.min(m*aB0, m*gEf(i+1,u)*stoppie); dec=Fb/m+kD*u*u/m+G*Crr*cosP[i]+G*sinP[i]; if(dec<0) dec=0; vb=Math.sqrt(u*u+2*dec*dsT); }
      return vb<vfi?vb:vfi;
    },
    tyre:function(i, v, k, along){                      // grip use on the friction circle (tyre forces incl. drag, rolling, grade)
      var gE=G*cosP[i]+v*v*kv[i]; if(gE<0.1) gE=0.1;
      var aT=along+kD*v*v/m+G*Crr*cosP[i]+G*sinP[i];
      return Math.hypot(v*v*Math.abs(k), aT)/(mu*gE);
    },
    gEff:function(i, v){ var gE=G*cosP[i]+v*v*kv[i]; return gE<0.1?0.1:gE; },
    topSpeed:function(){ var v=60; for(var it=0;it<60;it++){ var f=kD*v*v*v+m*G*Crr*v-Pw, df=3*kD*v*v+m*G*Crr; v-=f/df; if(v<1) v=1; } return v; }
  };
}

/* trajectory segment length from the integral: ds_traj = sqrt( (dc·(1 − n̄κ̄c))² + Δn² ) */
function segLen(g, n, i){
  var nb=0.5*(n[i]+n[i+1]), kb=0.5*(g.kc[i]+g.kc[i+1]), f=1-nb*kb; if(f<0.05) f=0.05;
  var a=g.dc[i]*f, b=n[i+1]-n[i]; return Math.sqrt(a*a+b*b);
}

/* full evaluation of a line n(s): geometry, speed profile, time and diagnostics */
function evalLine(g, n, inp, p){
  var N=g.N, ds=g.ds, i, M=model(g,inp,p);
  var Rx=new Float64Array(N), Ry=new Float64Array(N), dsT=new Float64Array(N), sR=new Float64Array(N);
  for(i=0;i<N;i++){ Rx[i]=g.Cx[i]+n[i]*g.Nx[i]; Ry[i]=g.Cy[i]+n[i]*g.Ny[i]; }
  for(i=0;i<N-1;i++){ dsT[i]=segLen(g,n,i); sR[i+1]=sR[i]+dsT[i]; }
  var j=Math.max(1,Math.round(4/ds)), kR=new Float64Array(N);
  for(i=0;i<N;i++){ var a=Math.max(0,i-j), b=Math.min(N-1,i+j); kR[i]=(b-a>=2)?circleCurv(Rx[a],Ry[a],Rx[i],Ry[i],Rx[b],Ry[b]):0; }
  var kw=gauss(kR,p.sigmaKappa/ds);                     // Gaussian on κ: progressive roll into the lean
  var vmax=new Float64Array(N); for(i=0;i<N;i++) vmax[i]=M.vmax(i,kw[i]);
  var vf=new Float64Array(N), vb=new Float64Array(N);
  vf[0]=p.fromRest?0:Math.min(vmax[0],M.topSpeed());
  for(i=0;i<N-1;i++) vf[i+1]=M.fwd(vf[i],i,kw[i],kw[i+1],dsT[i],vmax[i+1]);
  vb[N-1]=p.stopEnd?0:vf[N-1];
  for(i=N-2;i>=0;i--) vb[i]=M.bwd(vb[i+1],i,kw[i],kw[i+1],dsT[i],vf[i]);
  var lean=new Float64Array(N), grip=new Float64Array(N), t=new Float64Array(N), aLat=new Float64Array(N), aLong=new Float64Array(N), limit=new Array(N);
  for(i=0;i<N;i++){
    var w=vb[i], gE=M.gEff(i,w);
    aLat[i]=w*w*kw[i];
    aLong[i]= i<N-1 ? (vb[i+1]*vb[i+1]-w*w)/(2*dsT[i]) : (i>0?aLong[i-1]:0);
    lean[i]=Math.atan(w*w*Math.abs(kw[i])/gE)*180/Math.PI;                  // θ = arctan(V²κ / g)
    grip[i]=M.tyre(i,w,kw[i],aLong[i]);
    limit[i] = (w>=vmax[i]-0.05) ? M.vtype(i,kw[i]) : (vb[i]<vf[i]-0.05 ? 'brake' : 'accel');
    if(i<N-1){ var sv=w+vb[i+1]; t[i+1]=t[i]+2*dsT[i]/(sv>1e-3?sv:1e-3); }   // exact for constant acceleration
  }
  return {N:N, Cx:g.Cx, Cy:g.Cy, Nx:g.Nx, Ny:g.Ny, n:Float64Array.from(n), Rx:Rx, Ry:Ry, sR:sR, dsT:dsT, kc:g.kc, k:kw,
          v:vb, vmax:vmax, lean:lean, grip:grip, aLat:aLat, aLong:aLong, t:t, limit:limit, phi:g.phi, kv:g.kv,
          latCoef:M.latCoef, time:t[N-1]};
}

/* ---------- candidate lines ---------- */
function centreOffsets(g, inp){ var n=new Float64Array(g.N); for(var i=0;i<g.N;i++) n[i]=0.5*(inp.lo[i]+inp.hi[i]); return n; }
/* n = M·tanh(κ_lookahead·C), centred in the corridor */
function tanhOffsets(g, inp, p){
  var kl=gauss(g.kc,p.sigmaLook/g.ds), n=new Float64Array(g.N);
  for(var i=0;i<g.N;i++){ var mid=0.5*(inp.lo[i]+inp.hi[i]), M=0.5*(inp.hi[i]-inp.lo[i]); n[i]=mid+M*Math.tanh(kl[i]*p.tanhC); }
  return n;
}
/* Σ w·|P[i−1] − 2P[i] + P[i+1]|² over the corridor; pw = 2 → minimum curvature, pw > 2 → widest radius (IRLS).
   Projected Gauss–Seidel with over-relaxation, coarse-to-fine. */
function minCurvOffsets(g, inp, pw){
  var Cx=g.Cx, Cy=g.Cy, Nx=g.Nx, Ny=g.Ny, lo=inp.lo, hi=inp.hi, ds=g.ds;
  pw=pw||2;
  var N=Cx.length, n=centreOffsets(g,inp), i;
  var steps=[160,80,40,20,10,5].map(function(L){ return Math.round(L/ds); }).filter(function(s){ return s>=2; });
  steps.push(1);
  var seen={}; steps=steps.filter(function(s){ if(seen[s]) return false; seen[s]=1; return true; });
  for(var si=0; si<steps.length; si++){
    var step=steps[si], idx=[];
    for(i=0;i<N;i+=step) idx.push(i);
    if(idx[idx.length-1]!==N-1) idx.push(N-1);
    var m=idx.length; if(m<5) continue;
    var cx=new Float64Array(m), cy=new Float64Array(m), nx=new Float64Array(m), ny=new Float64Array(m),
        l=new Float64Array(m), h=new Float64Array(m), v=new Float64Array(m), px=new Float64Array(m), py=new Float64Array(m),
        w=new Float64Array(m).fill(1);
    for(var k=0;k<m;k++){ var q=idx[k]; cx[k]=Cx[q]; cy[k]=Cy[q]; nx[k]=Nx[q]; ny[k]=Ny[q]; l[k]=lo[q]; h[k]=hi[q]; v[k]=n[q];
      px[k]=cx[k]+v[k]*nx[k]; py[k]=cy[k]+v[k]*ny[k]; }
    var hstep=step*ds, outer=(pw>2 && hstep<=12)?10:1, sweeps=step===1?120:500, omega=1.5;
    for(var o=0;o<outer;o++){
      if(pw>2){
        var mx=0, raw=new Float64Array(m);
        for(i=1;i<m-1;i++) raw[i]=Math.hypot(px[i-1]-2*px[i]+px[i+1], py[i-1]-2*py[i]+py[i+1])/(hstep*hstep);
        raw[0]=raw[1]; raw[m-1]=raw[m-2];
        var rs=gauss(raw,15/hstep);
        for(i=0;i<m;i++){ w[i]=Math.pow(rs[i]*rs[i]+1e-8,(pw-2)/2); if(w[i]>mx) mx=w[i]; }
        for(i=0;i<m;i++) w[i]=w[i]/mx+1e-4;
      }
      var sw=Math.ceil(sweeps/(outer>1?2:1));
      for(var it=0; it<sw; it++){
        for(i=1;i<m-1;i++){
          var Ax=px[i-1]+px[i+1]-2*cx[i], Ay=py[i-1]+py[i+1]-2*cy[i];
          var num=2*w[i]*(Ax*nx[i]+Ay*ny[i]), den=4*w[i];
          if(i>=2){ var Bx=px[i-2]-2*px[i-1]+cx[i], By=py[i-2]-2*py[i-1]+cy[i]; num-=w[i-1]*(Bx*nx[i]+By*ny[i]); den+=w[i-1]; }
          if(i<=m-3){ var Dx=cx[i]-2*px[i+1]+px[i+2], Dy=cy[i]-2*py[i+1]+py[i+2]; num-=w[i+1]*(Dx*nx[i]+Dy*ny[i]); den+=w[i+1]; }
          var tt=v[i]+omega*(num/den-v[i]); if(tt<l[i]) tt=l[i]; else if(tt>h[i]) tt=h[i];
          v[i]=tt; px[i]=cx[i]+tt*nx[i]; py[i]=cy[i]+tt*ny[i];
        }
      }
    }
    for(k=0;k<m-1;k++){ var a=idx[k], b=idx[k+1]; for(i=a;i<=b;i++){ var f=(i-a)/((b-a)||1); n[i]=v[k]*(1-f)+v[k+1]*f; } }
  }
  for(i=0;i<N;i++){ if(n[i]<lo[i]) n[i]=lo[i]; else if(n[i]>hi[i]) n[i]=hi[i]; }
  return n;
}

/* ---------- minimum-time optimiser ----------
   Direct search on n(s) with smooth raised-cosine bumps. Each trial is evaluated EXACTLY but incrementally:
   only the curvature window around the bump is recomputed, the forward pass runs until it re-joins the previous
   speed profile and the backward pass likewise, so a trial costs a few hundred points instead of the whole road. */
function createOptimizer(g, n0, inp, p){
  var N=g.N, ds=g.ds, M=model(g,inp,p), lo=inp.lo, hi=inp.hi, i;
  var J=Math.max(1,Math.round(4/ds)), K=gaussKernel(p.sigmaKappa/ds), RG=K.R, ker=K.k;
  var names=['n','Rx','Ry','kR','kw','dsT','vmax','vf','vb'], A={}, S={};
  names.forEach(function(k){ A[k]=new Float64Array(N); S[k]=new Float64Array(N); });
  A.n.set(n0);
  for(i=0;i<N;i++){ A.Rx[i]=g.Cx[i]+A.n[i]*g.Nx[i]; A.Ry[i]=g.Cy[i]+A.n[i]*g.Ny[i]; }
  function kRat(Rx,Ry,i){ var a=Math.max(0,i-J), b=Math.min(N-1,i+J); return (b-a>=2)?circleCurv(Rx[a],Ry[a],Rx[i],Ry[i],Rx[b],Ry[b]):0; }
  function kwat(kR,i){ var acc=0; for(var q=-RG;q<=RG;q++){ var t=i+q; if(t<0) t=0; else if(t>=N) t=N-1; acc+=kR[t]*ker[q+RG]; } return acc; }
  for(i=0;i<N;i++) A.kR[i]=kRat(A.Rx,A.Ry,i);
  for(i=0;i<N;i++){ A.kw[i]=kwat(A.kR,i); A.vmax[i]=M.vmax(i,A.kw[i]); }
  for(i=0;i<N-1;i++) A.dsT[i]=segLen(g,A.n,i);
  A.vf[0]=p.fromRest?0:Math.min(A.vmax[0],M.topSpeed());
  for(i=0;i<N-1;i++) A.vf[i+1]=M.fwd(A.vf[i],i,A.kw[i],A.kw[i+1],A.dsT[i],A.vmax[i+1]);
  A.vb[N-1]=p.stopEnd?0:A.vf[N-1];
  for(i=N-2;i>=0;i--) A.vb[i]=M.bwd(A.vb[i+1],i,A.kw[i],A.kw[i+1],A.dsT[i],A.vf[i]);
  function dt(d, v1, v2){ var s=v1+v2; return 2*d/(s>1e-3?s:1e-3); }
  var T=0; for(i=0;i<N-1;i++) T+=dt(A.dsT[i],A.vb[i],A.vb[i+1]);
  names.forEach(function(k){ S[k].set(A[k]); });
  var T0=T, trials=0, accepted=0;

  function copy(from,to,a,b){ for(var q=0;q<names.length;q++){ var k=names[q]; to[k].set(from[k].subarray(a,b+1),a); } }

  /* try n += delta·bump(c, hw); keep it only if the time integral drops */
  function trial(c, hw, delta){
    trials++;
    var a=Math.max(1,c-hw), b=Math.min(N-2,c+hw), changed=false, q;
    for(q=a;q<=b;q++){ var x=(q-c)/hw, f=0.5*(1+Math.cos(Math.PI*x)), v=A.n[q]+delta*f;
      if(v<lo[q]) v=lo[q]; else if(v>hi[q]) v=hi[q]; S.n[q]=v; if(v!==A.n[q]) changed=true; }
    if(!changed) return 0;
    for(q=a;q<=b;q++){ S.Rx[q]=g.Cx[q]+S.n[q]*g.Nx[q]; S.Ry[q]=g.Cy[q]+S.n[q]*g.Ny[q]; }
    var ka=Math.max(0,a-J), kb=Math.min(N-1,b+J);
    for(q=ka;q<=kb;q++) S.kR[q]=kRat(S.Rx,S.Ry,q);
    var wa=Math.max(0,ka-RG), wb=Math.min(N-1,kb+RG);
    for(q=wa;q<=wb;q++){ S.kw[q]=kwat(S.kR,q); S.vmax[q]=M.vmax(q,S.kw[q]); }
    for(q=a-1;q<=b;q++) S.dsT[q]=segLen(g,S.n,q);
    var lowGeo=Math.min(wa,a-1), hiGeo=Math.max(wb,b);
    var p0=Math.max(0,lowGeo-1), fEnd=p0;
    for(q=p0;q<N-1;q++){
      S.vf[q+1]=M.fwd(S.vf[q],q,S.kw[q],S.kw[q+1],S.dsT[q],S.vmax[q+1]); fEnd=q+1;
      if(q+1>hiGeo && Math.abs(S.vf[q+1]-A.vf[q+1])<1e-9) break;
    }
    var q0;
    if(fEnd>=N-1){ q0=N-1; S.vb[N-1]=p.stopEnd?0:S.vf[N-1]; } else q0=fEnd+1;
    var bStart=q0;
    for(q=q0-1;q>=0;q--){
      S.vb[q]=M.bwd(S.vb[q+1],q,S.kw[q],S.kw[q+1],S.dsT[q],S.vf[q]); bStart=q;
      if(q<p0 && Math.abs(S.vb[q]-A.vb[q])<1e-9) break;
    }
    var s0=Math.max(0,Math.min(bStart,a-1)), s1=Math.min(N-2,Math.max(q0,b)), dT=0;
    for(q=s0;q<=s1;q++) dT+=dt(S.dsT[q],S.vb[q],S.vb[q+1])-dt(A.dsT[q],A.vb[q],A.vb[q+1]);
    var r0=Math.max(0,Math.min(wa,p0,bStart,a-1)), r1=Math.min(N-1,Math.max(wb,fEnd,q0,b));
    if(dT<-1e-7){ copy(S,A,r0,r1); T+=dT; accepted++; return dT; }
    copy(A,S,r0,r1); return 0;
  }

  // schedule: wide bumps first, then narrower; large steps first, then smaller
  var widths=[150,80,40,20,10].map(function(w){ return Math.round(w/ds); }).filter(function(h){ return h>=2; });
  if(!widths.length) widths=[2];
  var amps=[1.0,0.4,0.15,0.05], stages=2*widths.length*amps.length;
  var gen=(function*(){
    for(var round=0; round<2; round++){
      for(var wi=0; wi<widths.length; wi++){
        var hw=widths[wi], stride=Math.max(1,Math.round(hw/2));
        for(var ai=0; ai<amps.length; ai++){
          var amp=amps[ai], stage=round*widths.length*amps.length+wi*amps.length+ai;
          for(var sweep=0; sweep<3; sweep++){
            var before=T, k=0;
            for(var c=1; c<N-1; c+=stride){
              if(trial(c,hw,amp)<0){ for(var r=0;r<3 && trial(c,hw,amp)<0;r++); }
              else if(trial(c,hw,-amp)<0){ for(var r2=0;r2<3 && trial(c,hw,-amp)<0;r2++); }
              if(++k%40===0) yield (stage+(sweep+c/N)/3)/stages;
            }
            if(before-T<1e-4*Math.max(1,T/60)) break;
          }
        }
      }
    }
  })();

  return {
    T0:T0,
    get T(){ return T; },
    get n(){ return A.n; },
    get stats(){ return {trials:trials, accepted:accepted}; },
    step:function(budgetMs){                            // run for about budgetMs → {done, T, frac}
      var t0=Date.now(), r={done:false, frac:0};
      while(Date.now()-t0<budgetMs){ var x=gen.next(); if(x.done){ r.done=true; r.frac=1; break; } r.frac=x.value; }
      r.T=T; return r;
    }
  };
}

/* candidate lines: start points for the optimiser and rows of the comparison table */
function candidates(g, inp, p){
  var c={ centre:centreOffsets(g,inp), tanh:tanhOffsets(g,inp,p), mincurv:minCurvOffsets(g,inp,2), minmax:minCurvOffsets(g,inp,6) };
  var out={}; Object.keys(c).forEach(function(k){ out[k]={n:c[k], time:evalLine(g,c[k],inp,p).time}; });
  return out;
}

/* constraint report for a finished line */
function checkConstraints(r, inp){
  var maxOut=0, maxLean=0, maxGrip=0;
  for(var i=0;i<r.N;i++){
    var out=Math.max(inp.lo[i]-r.n[i], r.n[i]-inp.hi[i], 0); if(out>maxOut) maxOut=out;
    if(r.lean[i]>maxLean) maxLean=r.lean[i];
    if(r.grip[i]>maxGrip) maxGrip=r.grip[i];
  }
  return {maxOut:maxOut, maxLean:maxLean, leanLimit:Math.atan(r.latCoef)*180/Math.PI, maxGrip:maxGrip};
}

return {G:G, gauss:gauss, deriv:deriv, curvature:curvature, circleCurv:circleCurv, prepare:prepare, model:model, segLen:segLen,
        evalLine:evalLine, centreOffsets:centreOffsets, tanhOffsets:tanhOffsets, minCurvOffsets:minCurvOffsets,
        createOptimizer:createOptimizer, candidates:candidates, checkConstraints:checkConstraints};
})();
if(typeof module!=='undefined') module.exports=MotoSim;
