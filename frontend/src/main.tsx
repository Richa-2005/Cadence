import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { ArrowUpRight, ArrowRight, Fingerprint, Keyboard, Activity, Check, RotateCcw, ChevronDown, ShieldCheck, Plus, Trash2, CircleHelp, Shuffle, X, MousePointer2 } from 'lucide-react';
import './style.css';

type KeyEvent = { press: number; release: number };
type Attempt = { text: string; events: KeyEvent[] };
type Profile = { id: number; name: string; phrase: string };
type Result = { distance: number; accepted: boolean; threshold: number; rhythm: number[]; reference_rhythm: number[]; note: string };
const PHRASE = 'the rhythm is yours';
async function api<T>(path: string, data?: unknown): Promise<T> {
  const response = await fetch('/api/' + path, data ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) } : undefined);
  if (!response.ok) { const e = await response.json().catch(() => ({})); throw new Error(typeof e.detail === 'string' ? e.detail : 'Could not complete this request. Please try again.'); }
  return response.json();
}
function Bars({ values, muted = false }: { values?: number[]; muted?: boolean }) {
  const data = values?.length ? values : Array.from({ length: 44 }, (_, i) => .03 + ((i * 17 + 9) % 31) / 240);
  return <div className={'bars ' + (muted ? 'muted' : '')} aria-hidden="true">{data.map((v, i) => <i key={i} style={{ height: `${Math.max(8, Math.min(100, v * 490))}%` }} />)}</div>;
}
const PHRASES = ['the rhythm is yours', 'every keystroke tells a story', 'my fingers find their own rhythm', 'small pauses make a signature', 'a familiar rhythm feels like home'];
function KeystrokeArt() {
  const [key, setKey] = useState(0);
  useEffect(() => { const timer = setInterval(() => setKey(k => (k + 1) % 6), 850); return () => clearInterval(timer); }, []);
  return <div className="keystroke-art">
    <div className="art-title"><Keyboard size={22}/><span>Identity, between the keystrokes</span></div>
    <div className="finger-trails">{'rhythm'.split('').map((letter,i)=><div key={i} className={key===i?'striking':''}><span className="finger-path"/><kbd>{letter}</kbd></div>)}</div>
    <div className="key-art-metrics"><span><b>Press</b> how long you hold</span><span><b>Pause</b> when you move next</span></div>
    <div className="identity-flow"><span><Keyboard size={23}/>Keystrokes</span><ArrowRight size={18}/><span><Activity size={23}/>Timing patterns</span><ArrowRight size={18}/><span><Fingerprint size={25}/>Your identity</span></div>
    <p>Cadence reads the rhythm of your fingers, not a fingerprint scan.</p><span className="art-disclaimer">Animated illustration · not a live reading</span>
  </div>;
}
function TypingComparison() {
  const [playing, setPlaying] = useState(true), [tick, setTick] = useState(0);
  useEffect(()=>{if(!playing)return;const timer=setInterval(()=>setTick(t=>(t+1)%8),600);return()=>clearInterval(timer);},[playing]);
  return <div className="typing-explainer"><div className="comparison-heading"><h3>Watch two people type “rhythm”</h3><button onClick={()=>setPlaying(!playing)}>{playing?'Pause animation':'Play animation'}</button></div>
    <p>Same six keys. Different holds and pauses.</p>{['A','B'].map((person,row)=><div className={'typing-person person-'+person} key={person}><div className="person-heading"><b>Person {person}</b><span>{row===0?'Quick taps, longer pauses':'Longer holds, overlapping keys'}</span></div><div className="key-timeline">{'rhythm'.split('').map((letter,i)=><div key={i} className={(tick===(row===0?i:(i+2)%6)&&playing)?'key-active':''}><kbd>{letter}</kbd><span className="hold-block" style={{width:row===0?`${35+(i*13)%25}%`:`${65+(i*7)%30}%`}}/><small>{row===0?[65,82,71,94,60,88][i]:[135,148,122,165,141,130][i]} ms</small></div>)}</div></div>)}
    <div className="timeline-legend"><span><i/>Bar length = time holding a key</span><span>ms = milliseconds</span></div><p className="caption">Illustrative timings, not model predictions. Cadence combines key holds and the timing between keys.</p></div>;
}
const GUIDE = [
  {title:'Create your reference',text:'Pick a phrase and type it five times. We combine those attempts into your reference rhythm.',detail:'Choose a name, pick your phrase, then save five natural attempts.',icon:Fingerprint},
  {title:'Type as yourself',text:'Select your profile and type its phrase again. There is no need to copy your earlier speed.',detail:'We capture each press, release and overlap. Retry if you lose focus or make an unusual attempt.',icon:Keyboard},
  {title:'See the decision',text:'Get Accepted or Not accepted, with the difference from your reference explained underneath.',detail:'The backend applies a fixed cutoff from development experiments. Users cannot change the decision policy.',icon:ShieldCheck},
];
function GettingStarted(){const [step,setStep]=useState(0);const Icon=GUIDE[step].icon;return <><div className="guide-tabs" role="tablist" aria-label="How to use Cadence">{GUIDE.map((g,i)=><button role="tab" id={'guide-tab-'+i} aria-controls="guide-panel" aria-selected={step===i} key={g.title} onClick={()=>setStep(i)} className={step===i?'selected':''}><span>0{i+1}</span>{g.title}<ArrowUpRight size={18}/></button>)}</div><div className="guide-panel" id="guide-panel" role="tabpanel" aria-labelledby={'guide-tab-'+step}><div className="guide-animation" key={step}><Icon size={70} strokeWidth={1}/><div className="guide-dots">{Array.from({length:5},(_,i)=><span key={i} style={{animationDelay:`${i*.2}s`}}/>)}</div></div><div><h3>{GUIDE[step].title}</h3><p>{GUIDE[step].text}</p><p className="guide-detail">{GUIDE[step].detail}</p><a href="#demo" className="inline-link">{step===0?'Build my reference':'Open the typing workspace'}<ArrowRight size={18}/></a></div></div></>;}
function App() {
  const [ready, setReady] = useState(false), [status, setStatus] = useState('Connecting to local model…');
  const [profiles, setProfiles] = useState<Profile[]>([]), [selected, setSelected] = useState('');
  const [mode, setMode] = useState<'enrol' | 'verify'>('enrol'), [name, setName] = useState('');
  const [attempts, setAttempts] = useState<Attempt[]>([]), [text, setText] = useState(''), [events, setEvents] = useState<KeyEvent[]>([]);
  const [error, setError] = useState(''), [busy, setBusy] = useState(false), [result, setResult] = useState<Result | null>(null);
  const [notice, setNotice] = useState('');
  const [phrase, setPhrase] = useState(PHRASE), [replaceId, setReplaceId] = useState('');
  const activeProfile = profiles.find(p => String(p.id) === selected);
  const currentPhrase = mode === 'enrol' ? phrase : (activeProfile?.phrase || PHRASE);
  const active = useRef(new Map<string, number>()), completed = useRef<KeyEvent[]>([]), [held, setHeld] = useState(0);
  const input = useRef<HTMLInputElement>(null), results = useRef<HTMLElement>(null);
  useEffect(() => { api<{ready:boolean;model:string}>('health').then(h => { setReady(h.ready); setStatus(h.ready ? 'Local model connected' : 'Model assets unavailable'); }).catch(() => setStatus('Start the local API to try the demo')); api<Profile[]>('profiles').then(setProfiles).catch(() => {}); }, []);
  function reset() { active.current.clear(); completed.current = []; setEvents([]); setText(''); setHeld(0); setError(''); }
  function changeMode(next: 'enrol'|'verify') { reset(); setMode(next); setResult(null); setNotice(''); }
  function down(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.repeat) { e.preventDefault(); return; }
    if (e.ctrlKey || e.metaKey || e.altKey || ['Tab', 'Enter', 'Escape'].includes(e.key)) return;
    if (!active.current.has(e.code)) active.current.set(e.code, performance.now());
    setHeld(active.current.size);
  }
  function up(e: React.KeyboardEvent<HTMLInputElement>) {
    const press = active.current.get(e.code);
    if (press === undefined) return;
    completed.current.push({press, release:performance.now()}); active.current.delete(e.code);
    setEvents([...completed.current]); setHeld(active.current.size);
  }
  function blur() { if (active.current.size) { reset(); setError('Capture was interrupted while a key was held. Please retry.'); } }
  async function submit() {
    setError(''); setNotice('');
    if (text !== currentPhrase) { setError('Type the complete phrase exactly as shown.'); input.current?.focus(); return; }
    if (active.current.size || events.length < currentPhrase.length) { setError('Release all keys before saving your attempt.'); return; }
    if (mode === 'enrol' && (phrase.length < 12 || phrase.length > 60 || phrase.trim() !== phrase || !/^[\x20-\x7E]+$/.test(phrase))) { setError('Use a phrase of 12–60 keyboard characters, without spaces at either end.'); return; }
    if (mode === 'enrol' && !name.trim()) { setError('Add a profile name before saving your attempt.'); return; }
    const attempt = { text, events:[...events] };
    if (mode === 'enrol' && attempts.length < 4) { setAttempts([...attempts, attempt]); reset(); setNotice('Attempt saved. Take a breath, then type naturally again.'); input.current?.focus(); return; }
    setBusy(true);
    try {
      if (mode === 'enrol') {
        const profile = await api<Profile>('enrol', {name, phrase, replace_profile_id:replaceId ? Number(replaceId) : null, attempts:[...attempts, attempt]});
        setProfiles([profile, ...profiles.filter(p=>p.id!==profile.id)]); setResult(null); setSelected(String(profile.id)); setAttempts([]); setMode('verify'); reset(); setNotice('Your reference is ready. Type once more to compare your rhythm.');
      } else {
        if (!selected) throw new Error('Select an enrolled profile first.');
        const response = await api<Result>('verify', { profile_id:Number(selected), attempt });
        setResult(response); reset(); setTimeout(() => results.current?.scrollIntoView({behavior:'smooth',block:'start'}), 100);
      }
    } catch(e) { setError(e instanceof Error ? e.message : 'Request failed'); }
    finally { setBusy(false); }
  }
  const progress = Math.min(text.length / currentPhrase.length * 100, 100);
  return <>
    <header className="header wrap"><a className="brand" href="#"><span className="brand-icon"><Activity size={24}/></span>cadence<span className="brand-dot">.</span></a><div className="header-right"><span className="edition">A study in digital identity</span><a href="#demo" className="small-link">Try your rhythm <ArrowUpRight size={15}/></a></div></header>
    <main>
      <section className="hero wrap" id="about">
        <div className="hero-copy"><div className="eyebrow"><span/> Keystrokes. Finger patterns. The identity behind them.</div><h1>A rhythm<br/>of your <em>own.</em></h1><p className="hero-description">Your fingers have a familiar way of moving. Cadence learns the holds, pauses, and overlaps between your keystrokes to explore the identity behind your typing.</p><div className="hero-actions"><a href="#demo" className="button primary">Discover your rhythm <ArrowUpRight size={18}/></a><a href="#how" className="text-button">How it works <ArrowRight size={17}/></a></div><div className="hero-footnote"><Keyboard size={15}/> Best experienced on a physical keyboard</div></div>
        <KeystrokeArt/>
      </section>
      <section className="intro-band"><div className="wrap band-inner"><span className="band-label">Beyond the words</span><p>A password is <span>what you know.</span><br/>Your typing rhythm is <em>how you move.</em></p><span className="band-number">01 / THE IDEA</span></div></section>
      <section className="explain wrap"><div><div className="eyebrow">The pattern behind the keys</div><h2>Same sentence.<br/><em>Different rhythms.</em></h2><p>You might hold a key a little longer, overlap the next, or pause between words. Our neural encoder turns those timings into a compact signature, then compares it with your reference.</p><a href="#demo" className="inline-link">See what yours looks like <ArrowUpRight size={17}/></a></div><TypingComparison/></section>
      <section id="how" className="how-section"><div className="wrap"><div className="section-heading"><div><div className="eyebrow">Explore the three steps</div><h2>Make yourself <em>known.</em></h2></div><a href="#demo" className="button outline">Let's try it <ArrowUpRight size={17}/></a></div><GettingStarted/></div></section>
      <section id="demo" className="demo-section workspace-width"><div className="demo-title"><div className="eyebrow">Your typing workspace</div><h2>Let your fingers<br/><em>do the talking.</em></h2><p>One phrase. Your natural pace. A rhythm worth exploring.</p></div><div className="demo-shell"><div className="demo-body"><aside className="profile-panel"><span className="field-label">Your profile</span><div className="mode-tabs"><button onClick={()=>changeMode('enrol')} className={mode==='enrol'?'active':''}><Plus size={15}/>Enrol</button><button onClick={()=>changeMode('verify')} className={mode==='verify'?'active':''}><ShieldCheck size={15}/>Verify</button></div>{mode==='enrol'?<><label htmlFor="replace-reference">Create or replace a reference</label><select id="replace-reference" value={replaceId} disabled={busy} onChange={e=>{const id=e.target.value;setReplaceId(id);const p=profiles.find(p=>String(p.id)===id);setName(p?.name||'');setPhrase(p?.phrase||PHRASE);setAttempts([]);setResult(null);reset();}}><option value="">New profile</option>{profiles.map(p=><option key={p.id} value={p.id}>Replace {p.name}'s reference</option>)}</select>{replaceId&&<p className="helper">Your existing reference stays active until all five new attempts are saved.</p>}<label htmlFor="profile-name">What should we call you?</label><input id="profile-name" value={name} onChange={e=>setName(e.target.value)} placeholder="Your name or alias" maxLength={40}/><p className="helper">A local label, not an online account.</p><div className="enrol-progress"><span>YOUR REFERENCE <b>{attempts.length} / 5</b></span><div>{[0,1,2,3,4].map(i=><span key={i} className={i<attempts.length?'done':i===attempts.length?'current':''}>{i<attempts.length?<Check size={13}/>:i+1}</span>)}</div></div>{attempts.length>0&&<button className="quiet-button" onClick={()=>{setAttempts([]);reset();}}><Trash2 size={13}/>Discard saved attempts</button>}</>:<><label htmlFor="profile-choice">Compare against</label><select id="profile-choice" value={selected} onChange={e=>{setSelected(e.target.value);setResult(null);reset();}}><option value="">Select a profile</option>{profiles.map(p=><option key={p.id} value={p.id}>{p.name}</option>)}</select>{!profiles.length&&<p className="helper">Enrol five attempts to create your first reference.</p>}</>}
        {mode==='verify'&&selected&&<button className="button outline replace-button" onClick={()=>{setReplaceId(selected);setName(activeProfile?.name||'');setPhrase(activeProfile?.phrase||PHRASE);setAttempts([]);changeMode('enrol');}}>Change my reference<RotateCcw size={17}/></button>}<div className="privacy-note"><ShieldCheck size={19}/><p>Local by design.<br/><span>Your profile phrase, reference and timing preview stay on this computer.</span></p></div></aside>
        <div className="typing-panel">{mode==='enrol'&&<div className="phrase-settings"><label htmlFor="reference-phrase">Your reference phrase</label><div><input id="reference-phrase" value={phrase} minLength={12} maxLength={60} disabled={busy} onChange={e=>{setPhrase(e.target.value);setAttempts([]);setResult(null);reset();}}/><button className="button outline" disabled={busy} onClick={()=>{const choices=PHRASES.filter(p=>p!==phrase);setPhrase(choices[Math.floor(Math.random()*choices.length)]);setAttempts([]);setResult(null);reset();}}><Shuffle size={18}/>Random phrase</button></div><p>Use 12–60 characters. Changing the phrase starts five fresh attempts.</p></div>}<div className="typing-heading"><span className="pill">{mode==='enrol'?`ATTEMPT ${attempts.length+1} OF 5`:'A FRESH COMPARISON'}</span><span>Take your time. Be yourself.</span></div><label className="phrase-label" htmlFor="typing-input">Type this phrase</label><div className="target-phrase">{currentPhrase.split('').map((c,i)=><span key={i} className={i<text.length?(c===text[i]?'typed':'wrong'):''}>{c}</span>)}</div><div className="input-container"><input ref={input} id="typing-input" value={text} onChange={e=>setText(e.target.value)} onKeyDown={down} onKeyUp={up} onBlur={blur} onPaste={e=>{e.preventDefault();setError('Please type the phrase; pasting has no timing signature.');}} onDrop={e=>e.preventDefault()} onCompositionStart={()=>{reset();setError('Use direct physical-keyboard input for this demo.');}} autoComplete="off" autoCapitalize="off" spellCheck={false} placeholder="Start typing here…" maxLength={70} disabled={busy}/><div className="input-progress" style={{width:`${progress}%`}}/></div><div className="capture-info"><span><span className={'tiny-dot '+(held?'pulsing':'')}/>{held?'Capturing your keystrokes':events.length?'Rhythm captured · release keys before saving':'Your rhythm will appear as you type'}</span><span>{text.length} / {currentPhrase.length}</span></div><div className="live-rhythm"><span className="field-label">How long you hold each key</span>{events.length?<Bars values={events.map(e=>(e.release-e.press)/1000)}/>:<div className="empty-rhythm"><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/></div>}</div><div aria-live="polite">{error&&<p className="error">{error}</p>}{notice&&<p className="notice"><Check size={15}/>{notice}</p>}</div><div className="typing-actions"><button className="quiet-button" onClick={reset} disabled={busy}><RotateCcw size={14}/>Retry attempt</button><button className="button primary" onClick={submit} disabled={!ready||busy||(mode==='verify'&&!selected)}>{busy?'Comparing…':mode==='enrol'?(attempts.length===4?(replaceId?'Replace my reference':'Create my reference'):'Save attempt'):'Compare my rhythm'}<ArrowRight size={17}/></button></div></div></div></div><p className="workspace-note"><Keyboard size={18}/>Use a physical keyboard. Paste and held-key repeats are disabled. <span className={ready?'connection-ready':''}>{status}</span></p></section>
      <section className="results-section workspace-width" ref={results} id="results"><div className="results-heading"><span className="eyebrow">Your result</span><span>04 / YOUR RESULTS</span></div>{result?<div className={'result-card '+(result.accepted?'accepted':'rejected')}><div className="decision-block"><span className="field-label">Your verification result</span><div className="decision-icon">{result.accepted?<ShieldCheck size={42}/>:<X size={42}/>}</div><h3>{result.accepted?'Accepted':'Not accepted'}</h3><p>{result.accepted?'Your typing was close enough to your saved reference for this demo.':'Your typing differed more than the demo allows. Try again at your natural pace.'}</p><div className="experimental-banner">Fixed cutoff from CMU development experiments. Not calibrated for live browser typing.</div><a className="button outline" href="#demo">Try another attempt<RotateCcw size={18}/></a></div><div className="rhythm-comparison"><h3>How we reached this result</h3><p className="score-explainer">We compare your new typing pattern with the average of your five reference attempts. The number below measures how different those patterns are: <b>0 means identical, and smaller means more alike.</b> It is not a percentage or a probability.</p><div className="score-numbers"><div><span>Your pattern difference</span><strong>{result.distance.toFixed(4)}</strong></div><div><span>Backend acceptance cutoff</span><strong>{result.threshold.toFixed(4)}</strong></div></div>{result.threshold!==null&&<div className="decision-equation">{result.distance.toFixed(4)} {result.accepted?'≤':'>'} {result.threshold.toFixed(4)} → {result.accepted?'Accepted':'Not accepted'} under the fixed backend policy</div>}<div className="difference-meter"><span style={{left:`${Math.min(result.distance/2*100,100)}%`}} title="Your difference"/>{result.threshold!==null&&<i style={{left:`${result.threshold/2*100}%`}} title="Cutoff"/>}</div><div className="meter-labels"><span>0 · Identical pattern</span><span>2 · Very different pattern</span></div><details className="result-traces"><summary>See the keystroke timing comparison<ChevronDown size={18}/></summary><div className="rhythm-label">One saved enrolment attempt <span>key hold times</span></div><Bars values={result.reference_rhythm} muted/><div className="rhythm-label">Your latest attempt <span>key hold times</span></div><Bars values={result.rhythm}/><p className="caption">These bars preview key holds. The decision also uses pauses and overlaps, compared against the averaged five-attempt reference.</p></details></div></div>:<div className="results-placeholder"><Activity size={28} strokeWidth={1.3}/><div><h3>Your next attempt has a story to tell.</h3><p>Enrol your reference, then verify to see your comparison here.</p></div><a href="#demo" aria-label="Go to typing demo"><ArrowUpRight size={24}/></a></div>}
      <details className="technical"><summary>Under the rhythm <span>Model details & limitations <ChevronDown size={16}/></span></summary><div><p>The v1 encoder uses a 1D CNN and a packed LSTM to turn up to 70 keystroke events into a 64-dimensional embedding. Five embeddings are averaged and normalised for enrolment.</p><p>Evaluated five-sample test EER: approximately 8.3% on Aalto and 18.5% on CMU. These datasets and protocols differ from live browser input. No live authentication threshold has been calibrated. This is an experimental demo, not a production security system.</p><p>The local service uses model seed 123 as an explicit demo configuration. Profiles are stored in SQLite on this machine; clear local data by removing the profile database while the API is stopped.</p></div></details></section>
      <section className="closing wrap"><span className="closing-mark"><Activity size={32}/></span><p>A small pause.<br/><em>A little more you.</em></p><a href="#demo" className="inline-link">Try another rhythm <ArrowUpRight size={16}/></a></section>
    </main><footer className="wrap"><a className="brand" href="#">cadence<span className="brand-dot">.</span></a><span>An exploration of identity, one keystroke at a time.</span><span>Built with curiosity. Tested with care.</span></footer>
  </>;
}
createRoot(document.getElementById('root')!).render(<React.StrictMode><App/></React.StrictMode>);
