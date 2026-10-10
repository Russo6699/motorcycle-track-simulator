/* SatCorridor — finds the road in satellite imagery along a marked route and returns a dense, smooth copy of it.
   The imagery is sampled "unrolled" along the route: row r = station every rowStep metres along the route
   (smooth spline through the route vertices), column q = lateral offset (q − (cols−1)/2)·colRes metres,
   positive = left of travel. The image analysis itself (OpenCV + scipy splprep) is roadfit.py, run in the
   browser with Pyodide. Long routes are processed in overlapping sections to bound memory. */
var SatCorridor = (function(){
"use strict";
var TS=256, PYODIDE='https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
var IMAGERY='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}';
function project(lat,lon,z){ var n=Math.pow(2,z)*TS, s=Math.sin(lat*Math.PI/180); return [(lon+180)/360*n, (0.5-Math.log((1+s)/(1-s))/(4*Math.PI))*n]; }

/* stations along the route. poly: [[lat,lon],...] */
function stations(poly, opt){
  opt=opt||{};
  var rowStep=opt.rowStep||1.0, colRes=opt.colRes||0.25, half=opt.halfWidth||28;
  var lat0=poly[0][0], lon0=poly[0][1], kx=Math.cos(lat0*Math.PI/180)*111320, ky=110540;
  var P=poly.map(function(q){ return [(q[1]-lon0)*kx,(q[0]-lat0)*ky]; });
  var SP=MotoSim.splineResample(P,rowStep), N=SP.N, X=SP.X, Y=SP.Y;
  var Nx=new Float64Array(N), Ny=new Float64Array(N), w=Math.max(1,Math.round(4/SP.ds)), i;   // normals over ±4 m
  for(i=0;i<N;i++){ var a=Math.max(0,i-w), b=Math.min(N-1,i+w), tx=X[b]-X[a], ty=Y[b]-Y[a], l=Math.hypot(tx,ty)||1; Nx[i]=-ty/l; Ny[i]=tx/l; }
  return {N:N, X:X, Y:Y, Nx:Nx, Ny:Ny, D:SP.D, rowStep:SP.ds, colRes:colRes, half:half, cols:Math.round(2*half/colRes)+1,
          zoom:opt.zoom||18, url:opt.url||IMAGERY, lat0:lat0, lon0:lon0, kx:kx, ky:ky};
}

/* imagery for rows r0..r1-1 → Uint8Array rows×cols×3 (bilinear) */
async function sample(st, r0, r1, onProg){
  var z=st.zoom, cols=st.cols, c0=(cols-1)/2, half=st.half, X=st.X, Y=st.Y, Nx=st.Nx, Ny=st.Ny, i, r, q;
  var need=new Map();
  function addTile(x,y){ var tx=Math.floor(x/TS), ty=Math.floor(y/TS), k=tx+'/'+ty; if(!need.has(k)) need.set(k,[tx,ty]); }
  for(i=r0;i<r1;i+=2){ for(var o=-half;o<=half;o+=half/4){ var g=project(st.lat0+(Y[i]+o*Ny[i])/st.ky, st.lon0+(X[i]+o*Nx[i])/st.kx, z);
    addTile(g[0]-1,g[1]-1); addTile(g[0]+1,g[1]+1); } }
  var keys=Array.from(need.keys()), tiles=new Map(), done=0;
  var cv=document.createElement('canvas'); cv.width=cv.height=TS; var cx=cv.getContext('2d',{willReadFrequently:true});
  async function worker(){
    while(keys.length){
      var k=keys.pop(), t=need.get(k), u=st.url.replace('{z}',z).replace('{x}',t[0]).replace('{y}',t[1]), res=null;
      for(var tries=0;tries<3 && !res;tries++){
        try{ var rr=await fetch(u); if(rr.ok) res=rr; else if(rr.status<500) break; }catch(e){}
        if(!res) await new Promise(function(f){ setTimeout(f,600*(tries+1)); });
      }
      if(!res) throw new Error('אריח של תצלום הלוויין לא נטען ('+k+')');
      var bmp=await createImageBitmap(await res.blob()); cx.clearRect(0,0,TS,TS); cx.drawImage(bmp,0,0);
      tiles.set(k, cx.getImageData(0,0,TS,TS).data); done++; if(onProg) onProg(done/need.size);
    }
  }
  await Promise.all([worker(),worker(),worker(),worker(),worker(),worker()]);
  var rows=r1-r0, img=new Uint8Array(rows*cols*3);
  function px(gx,gy,ch){ var tx=Math.floor(gx/TS), ty=Math.floor(gy/TS), d=tiles.get(tx+'/'+ty); if(!d) return 0; return d[((gy-ty*TS)*TS+(gx-tx*TS))*4+ch]; }
  for(r=0;r<rows;r++){
    var R=r0+r;
    for(q=0;q<cols;q++){
      var off=(q-c0)*st.colRes, G=project(st.lat0+(Y[R]+off*Ny[R])/st.ky, st.lon0+(X[R]+off*Nx[R])/st.kx, z);
      var fx=G[0]-0.5, fy=G[1]-0.5, ix=Math.floor(fx), iy=Math.floor(fy), ax=fx-ix, ay=fy-iy, k3=(r*cols+q)*3;
      for(var ch=0;ch<3;ch++){
        var v=px(ix,iy,ch)*(1-ax)*(1-ay)+px(ix+1,iy,ch)*ax*(1-ay)+px(ix,iy+1,ch)*(1-ax)*ay+px(ix+1,iy+1,ch)*ax*ay;
        img[k3+ch]=v<0?0:(v>255?255:v);
      }
    }
  }
  return {img:img, tiles:need.size};
}

/* Pyodide with numpy, scipy and OpenCV, loaded once per page */
var pyReady=null;
function loadPy(onMsg, roadfitSrc){
  if(pyReady) return pyReady;
  pyReady=(async function(){
    if(typeof loadPyodide==='undefined'){
      if(onMsg) onMsg('טוען את סביבת Python לדפדפן (Pyodide)…');
      await new Promise(function(res,rej){ var s=document.createElement('script'); s.src=PYODIDE+'pyodide.js'; s.onload=res; s.onerror=function(){ rej(new Error('טעינת Pyodide נכשלה')); }; document.head.appendChild(s); });
    }
    var py=await loadPyodide({indexURL:PYODIDE});
    if(onMsg) onMsg('טוען numpy, scipy ו־OpenCV (בפעם הראשונה כ־30 MB, אחר כך מהמטמון)…');
    await py.loadPackage(['numpy','scipy','opencv-python']);
    py.FS.writeFile('/home/pyodide/roadfit.py', roadfitSrc);
    py.runPython('import sys; sys.path.insert(0,"/home/pyodide"); import roadfit');
    return py;
  })();
  pyReady.catch(function(){ pyReady=null; });
  return pyReady;
}
function pyArr(o, k){ var a=o.get(k), b=a.getBuffer(), out=b.data.slice(); b.release(); a.destroy(); return out; }
function tick(){ return new Promise(function(r){ setTimeout(r,0); }); }

/* the whole step: poly [[lat,lon]] → {lat, lon, width, conf, kappa, D, stats, rows}.
   hooks: {msg(text), prog(0..1), roadfitSrc} */
async function run(poly, opt, hooks){
  opt=opt||{}; hooks=hooks||{};
  var msg=hooks.msg||function(){}, prog=hooks.prog||function(){};
  var st=stations(poly,opt), N=st.N, CH=opt.chunkRows||4000, OV=opt.overlapRows||300, outN=opt.outN||50000;
  var py=await loadPy(msg, hooks.roadfitSrc);
  // sections [a,b) of ≤CH rows, overlapping by OV
  var secs=[], a=0;
  while(true){ var b=Math.min(N,a+CH); if(N-b<OV) b=N; secs.push([a,b]); if(b>=N) break; a=b-OV; }
  var off=new Float64Array(N), raw=new Float64Array(N), hf=new Float64Array(N), rel=new Float64Array(N), wok=new Float64Array(N), wsum=new Float64Array(N);
  var nTiles=0, sigLab=null, many=secs.length>1;
  for(var s=0;s<secs.length;s++){
    var r0=secs[s][0], r1=secs[s][1], rows=r1-r0, base=s/secs.length, span=1/secs.length;
    msg('מוריד תצלום לוויין'+(many?' (קטע '+(s+1)+' מתוך '+secs.length+')':'')+'…');
    var smp=await sample(st, r0, r1, function(f){ prog(base+span*0.5*f); }); nTiles+=smp.tiles;
    msg('מאתר את הכביש בתצלום (OpenCV)'+(many?' – קטע '+(s+1)+' מתוך '+secs.length:'')+'…'); prog(base+span*0.55); await tick();
    py.globals.set('img_js', smp.img);
    var det=py.runPython('roadfit.detect_road(img_js.to_py(), '+rows+', '+st.cols+', '+st.rowStep+', '+st.colRes+')');
    py.globals.delete('img_js');
    var o=pyArr(det,'offset'), rw=pyArr(det,'off_raw'), h=pyArr(det,'half'), rl=pyArr(det,'rel'), wk=pyArr(det,'w_ok');
    var sl=det.get('sigma_lab'); sigLab=sl.toJs(); sl.destroy(); det.destroy();
    // blend into the full arrays; weights ramp across the overlaps so the sections join smoothly
    for(var r=0;r<rows;r++){
      var R=r0+r, wgt=1;
      if(s>0 && r<OV) wgt=Math.min(wgt,(r+0.5)/OV);
      if(s<secs.length-1 && r>=rows-OV) wgt=Math.min(wgt,(rows-r-0.5)/OV);
      off[R]+=wgt*o[r]; raw[R]+=wgt*rw[r]; hf[R]+=wgt*h[r]; rel[R]+=wgt*rl[r]; wok[R]+=wgt*wk[r]; wsum[R]+=wgt;
    }
    prog(base+span); await tick();
  }
  for(var i=0;i<N;i++){ var ww=wsum[i]||1; off[i]/=ww; raw[i]/=ww; hf[i]/=ww; rel[i]/=ww; wok[i]/=ww; }
  msg('מעביר עקומת B‑spline חלקה (scipy splprep) ודוגם '+outN.toLocaleString('en-US')+' נקודות…'); await tick();
  var G={X_js:st.X, Y_js:st.Y, Nx_js:st.Nx, Ny_js:st.Ny, D_js:st.D, off_js:off, raw_js:raw, half_js:hf, rel_js:rel, wok_js:wok};
  Object.keys(G).forEach(function(k){ py.globals.set(k,G[k]); });
  var fr=py.runPython('import numpy as np\n_a = [np.asarray(v.to_py(), dtype=float) for v in (X_js, Y_js, Nx_js, Ny_js, D_js, off_js, raw_js, half_js, rel_js, wok_js)]\n'+
    'roadfit.fit_centreline(*_a[:5], '+st.rowStep+', *_a[5:9], w_ok=_a[9], out_n='+outN+')');
  py.runPython('del _a');
  Object.keys(G).forEach(function(k){ py.globals.delete(k); });
  var x=pyArr(fr,'x'), y=pyArr(fr,'y'), out={width:pyArr(fr,'width'), conf:pyArr(fr,'conf'), kappa:pyArr(fr,'kappa'), D:pyArr(fr,'D')};
  var stp=fr.get('stats'); out.stats=stp.toJs({dict_converter:Object.fromEntries}); stp.destroy(); fr.destroy();
  out.stats.sigma_lab=sigLab; out.stats.tiles=nTiles; out.stats.sections=secs.length;
  out.lat=new Float64Array(x.length); out.lon=new Float64Array(x.length);
  for(var j=0;j<x.length;j++){ out.lat[j]=st.lat0+y[j]/st.ky; out.lon[j]=st.lon0+x[j]/st.kx; }
  // per station, for drawing the detected road edges on the map
  out.rows={X:st.X, Y:st.Y, Nx:st.Nx, Ny:st.Ny, offset:off, half:hf, rel:rel, lat0:st.lat0, lon0:st.lon0, kx:st.kx, ky:st.ky};
  prog(1);
  return out;
}

return {stations:stations, sample:sample, loadPy:loadPy, run:run, project:project};
})();
if(typeof module!=='undefined') module.exports=SatCorridor;
