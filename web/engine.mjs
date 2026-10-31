export function recommend(data, likedIds=[], excludedIds=[], limit=6, category='', query='') {
  const index=new Map(data.products.map((p,i)=>[p.id,i]));
  const liked=[...new Set(likedIds.filter(id=>index.has(id)))];
  const excluded=new Set([...liked,...excludedIds]);
  const profile=liked.map(id=>index.get(id));
  const c=data.config;
  const collaborative=data.products.map((_,i)=>Math.max(0,profile.reduce((s,j)=>s+data.ease[j][i],0)));
  const scale=Math.max(...collaborative)||1;
  return data.products.flatMap((p,i)=>{
    if(excluded.has(p.id)||(category&&p.category!==category)||!(p.name+' '+p.brand+' '+p.categories).toLowerCase().includes(query.toLowerCase()))return [];
    const base=data.popularity[i]*data.quality[i]**c.sentiment_power;
    const content=profile.length?profile.reduce((s,j)=>s+data.content[j][i],0)/profile.length:0;
    const score=Number((base+c.collaborative_weight*collaborative[i]/scale+c.content_weight*content).toFixed(8));
    let reason='Popular with reviewers';
    if(profile.length&&c.collaborative_weight>0&&collaborative[i]/scale>.1)reason='Related to your picks';
    else if(c.sentiment_power&&p.positive_fraction>.9)reason='Strong review sentiment';
    return [{...p,score,reason}];
  }).sort((a,b)=>b.score-a.score||(a.id<b.id?-1:a.id>b.id?1:0)).slice(0,limit);
}
