const {useState, useEffect} = React, h = React.createElement;
// Same origin when served by Flask; otherwise (e.g. VS Code Live Server) call Flask on port 5000.
const API = location.port === '5000' ? '' : 'http://localhost:5000';

const PRESETS = {
  fake:  {followers:'120',  following:'1800', posts:'3',   age:'20',   pic:true, url:true,  verified:false},
  real:  {followers:'850',  following:'410',  posts:'240', age:'1400', pic:true, url:false, verified:false},
  blank: {followers:'',     following:'',     posts:'',    age:'',     pic:true, url:false, verified:false}
};
const LABELS = {
  log_followers:'Followers', log_following:'Following', log_posts:'Posts', log_age_days:'Account age',
  follow_ratio_log:'Followers/following ratio', has_profile_pic:'Profile picture',
  has_external_url:'External URL', is_verified:'Verified status'
};

function Num({label, k, f, set, suffix}) {
  return h('label', {className:'f'}, label + (suffix ? ' (' + suffix + ')' : ''),
    h('input', {type:'number', min:0, inputMode:'numeric', value:f[k], onChange:e => set({...f, [k]:e.target.value})}));
}
function Check({label, k, f, set}) {
  return h('label', null, h('input', {type:'checkbox', checked:f[k], onChange:e => set({...f, [k]:e.target.checked})}), label);
}

function Result({r, info}) {
  if (!r) return h('p', {className:'empty'}, 'Enter the account details and select Check account, or load an example.');
  const pct = Math.round(r.probability_fake * 100);
  const imp = info ? Object.entries(info.feature_importance).sort((a, b) => b[1] - a[1]) : [];
  const max = imp.length ? imp[0][1] : 1;
  return h('div', null,
    h('p', {className:'verdict ' + (r.is_fake ? 'fake' : 'real')}, r.prediction),
    h('p', {className:'prob'}, pct + '% fake probability'),
    h('div', {className:'meter', role:'img', 'aria-label':'Fake probability ' + pct + ' percent'},
      h('div', {className:'pin', style:{left:pct + '%'}})),
    h('div', {className:'scale'}, h('span', null, 'Real'), h('span', null, 'Fake')),
    imp.length && h('div', null,
      h('h2', null, 'What the model relies on most'),
      h('ul', {className:'why'}, imp.map(([k, v]) =>
        h('li', {key:k},
          h('span', null, LABELS[k] || k),
          h('div', {className:'bar'}, h('i', {style:{width:(v / max * 100) + '%', background:'var(--ink)'}})))))),
    h('p', {className:'note'},
      'Prediction by ' + r.model + ' trained on labeled account data. This is a prediction, not proof that an account is fraudulent.'));
}

function App() {
  const [f, set] = useState(PRESETS.blank);
  const [r, setR] = useState(null);
  const [err, setErr] = useState('');
  const [busy, setBusy] = useState(false);
  const [info, setInfo] = useState(null);
  const [online, setOnline] = useState(null);

  useEffect(() => {
    fetch(API + '/model-info').then(x => x.json()).then(d => { setInfo(d); setOnline(true); }).catch(() => setOnline(false));
  }, []);

  const run = async (vals) => {
    const v = vals || f;
    const bad = ['followers', 'following', 'posts', 'age'].filter(k => v[k] === '' || isNaN(+v[k]) || +v[k] < 0);
    if (bad.length) { setErr('Enter a number of 0 or more for followers, following, posts and account age.'); setR(null); return; }
    setErr(''); setBusy(true);
    try {
      const res = await fetch(API + '/predict', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({followers:+v.followers, following:+v.following, posts:+v.posts, account_age_days:+v.age,
          has_profile_pic:v.pic ? 1 : 0, has_external_url:v.url ? 1 : 0, is_verified:v.verified ? 1 : 0})
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Request failed');
      setR(data); setOnline(true);
    } catch (e) {
      setR(null);
      setErr(e instanceof TypeError ? 'Cannot reach the API. Start it with: python app.py' : e.message);
      if (e instanceof TypeError) setOnline(false);
    }
    setBusy(false);
  };
  const load = k => { set(PRESETS[k]); setErr(''); setR(null); if (k !== 'blank') run(PRESETS[k]); };

  return h(React.Fragment, null,
    h('h1', null, 'Is this account real?'),
    h('p', {className:'lede'}, 'Enter an account\u2019s public numbers. A trained scikit-learn model estimates how likely it is to be fake.'),
    h('p', {className:'status'}, online === null ? 'Checking the model API...' :
      online ? h(React.Fragment, null, 'Model API connected: ', h('b', null, info ? info.model : '')) :
      'Model API not running. Start it with: python app.py'),
    h('div', {className:'grid'},
      h('section', {className:'panel'},
        h('h2', null, 'Account details'),
        h('div', {className:'presets'},
          h('button', {className:'chip', onClick:() => load('fake')}, 'Load suspicious example'),
          h('button', {className:'chip', onClick:() => load('real')}, 'Load normal example'),
          h('button', {className:'chip', onClick:() => load('blank')}, 'Clear')),
        h('div', {className:'fields'},
          h(Num, {label:'Followers', k:'followers', f, set}),
          h(Num, {label:'Following', k:'following', f, set}),
          h(Num, {label:'Posts', k:'posts', f, set}),
          h(Num, {label:'Account age', k:'age', f, set, suffix:'days'})),
        h('div', {className:'checks'},
          h(Check, {label:'Has a profile picture', k:'pic', f, set}),
          h(Check, {label:'Has an external URL in bio', k:'url', f, set}),
          h(Check, {label:'Verified', k:'verified', f, set})),
        h('button', {className:'go', disabled:busy, onClick:() => run()}, busy ? 'Checking...' : 'Check account'),
        h('p', {className:'err', role:'alert'}, err)),
      h('section', {className:'panel', 'aria-live':'polite'}, h('h2', null, 'Result'), h(Result, {r, info}))));
}
ReactDOM.createRoot(document.getElementById('root')).render(h(App));
