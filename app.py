"""DocQuery's Streamlit interface. Retrieval and generation live in existing modules."""
from contextlib import contextmanager
import re
import time
import warnings

import pandas as pd
import plotly.express as px
import streamlit as st

from config import ROOT
from ingest import build_index_from_pdfs, add_pdfs_to_existing_index, EMBEDDING_MODEL_NAME
from rag_engine import answer_question, retrieve, summarize_document, cross_document_query, GROK_MODEL, USE_HYBRID_SEARCH
from evaluation import summarize_retrieval, evaluate_answer_quality, logs_to_dataframe
from session_utils import stage_uploads
from citations import validate_citations

st.set_page_config(page_title='DocQuery', page_icon=':material/description:', layout='centered', initial_sidebar_state='expanded')
st.title('DocQuery')
st.write('Multi-document RAG with evaluated retrieval, evidence inspection, and page-level citations.')
st.caption('Evaluated retrieval · Page-level citations · Multi-document Q&A')

for name, default in {'store': None, 'chat_history': [], 'eval_logs': [], 'doc_names': [],
                      'last_sources': [], 'last_elapsed': 0.0, 'summary_result': None,
                      'cross_result': None, 'search_results': None, 'demo_active': False}.items():
    if name not in st.session_state:
        st.session_state[name] = default


def render_with_latex(text):
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    st.markdown(text)


@contextmanager
def processing(label, pinned=False):
    """Emit feedback before work starts and always clear it on exit."""
    if pinned:
        with st.bottom:
            placeholder = st.empty()
    else:
        placeholder = st.empty()
    with placeholder.container():
        status = st.status(label, expanded=False)
    try:
        yield status
    except Exception:
        status.update(label='Operation failed', state='error', expanded=False)
        raise
    else:
        status.update(label='Complete', state='complete', expanded=False)
    finally:
        placeholder.empty()


def reset_results():
    st.session_state.demo_active = False
    st.session_state.chat_history = []
    st.session_state.eval_logs = []
    st.session_state.last_sources = []
    st.session_state.last_elapsed = 0.0
    st.session_state.summary_result = None
    st.session_state.cross_result = None
    st.session_state.search_results = None
    st.session_state['chat_scope'] = []
    st.session_state['search_scope'] = []


def show_citation_status(validation):
    if not validation:
        return
    if validation['status'] == 'matched':
        st.caption('Citation references verified against retrieved evidence. This does not verify factual entailment or correctness.')
    elif validation['status'] == 'invalid':
        st.warning('Some citation references fall outside the retrieved evidence. Review the answer before relying on it.')
    elif validation['status'] == 'missing':
        st.warning('No recognized page citations. Inspect the retrieved evidence before relying on this response.')


def show_sources(sources):
    for rank, source in enumerate(sources, 1):
        with st.container(border=True):
            st.markdown(f"**{rank}. {source['doc_name']}**")
            extraction = 'OCR' if source.get('source') == 'ocr' else 'Native text'
            st.caption(f"Page {source['page']} · {extraction}")
            if source.get('text'):
                st.write(source['text'])
            strategy = 'Hybrid RRF' if source.get('score_type', 'rrf') == 'rrf' else 'Dense / FAISS'
            score_label = 'Rank score' if strategy == 'Hybrid RRF' else 'Cosine similarity'
            with st.popover('Retrieval details'):
                st.caption(f"Retrieval strategy: {strategy} · {score_label}: {source['score']:.4f}")
                st.caption('Ranking information, not an answer-confidence probability.')
                if source.get('chunk_id'):
                    st.code(source['chunk_id'], language=None)


def frozen_benchmark():
    """Read the authoritative first results table, never duplicate score constants."""
    text = (ROOT / 'docs' / 'evaluation.md').read_text(encoding='utf-8')
    size = re.search(r'\*\*Dataset:\*\* (\d+) documents.*?; (\d+) questions', text)
    methods = ('Dense / FAISS', 'BM25', 'Hybrid RRF')
    rows = {}
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
        if len(cells) >= 4 and cells[0] in methods and cells[0] not in rows:
            rows[cells[0]] = {'Method': cells[0], 'Hit@3': float(cells[1]), 'Recall@3': float(cells[2]), 'MRR@3': float(cells[3])}
    if not size or set(rows) != set(methods):
        raise ValueError('Frozen benchmark table is unavailable')
    return pd.DataFrame([rows[method] for method in methods]), int(size[2]), int(size[1])


with st.sidebar:
    st.header('DocQuery')
    st.caption('Evidence-grounded document intelligence')
    st.subheader('Upload PDFs')
    uploaded_files = st.file_uploader('Upload one or more PDFs', type=['pdf'], accept_multiple_files=True, label_visibility='collapsed')
    if st.button('Build / Update Index', type='primary', disabled=not uploaded_files, width='stretch'):
        try:
            with processing('Reading and indexing PDFs…'):
                with warnings.catch_warnings(record=True) as ingestion_warnings:
                    warnings.simplefilter('always', RuntimeWarning)
                    with stage_uploads(uploaded_files) as paths:
                        if st.session_state.store is None:
                            st.session_state.store = build_index_from_pdfs(paths, persist=False)
                        else:
                            st.session_state.store = add_pdfs_to_existing_index(paths, st.session_state.store, persist=False)
                        st.session_state.doc_names = st.session_state.store.list_documents()
                        reset_results()
            for warning in ingestion_warnings:
                st.warning(str(warning.message))
            st.success(f"Library updated: {len(st.session_state.doc_names)} documents · {len(st.session_state.store.chunks)} chunks")
        except (ValueError, OSError, RuntimeError) as exc:
            st.error(f'Could not index the upload: {exc}')
        except Exception:
            st.error('Could not index the upload. Check that the PDF is readable and the embedding model is available.')

    st.subheader('Document Library')
    if st.session_state.doc_names:
        st.caption(f"{len(st.session_state.doc_names)} documents · {len(st.session_state.store.chunks)} indexed chunks")
        selected = []
        for name in st.session_state.doc_names:
            count = sum(c.doc_name == name for c in st.session_state.store.chunks)
            if st.checkbox(name, key=f'select_{name}', help=f'{count} indexed chunks; select to remove'):
                selected.append(name)
        if selected and st.button(f'Remove selected ({len(selected)})', width='stretch'):
            for name in selected:
                st.session_state.store.remove_document(name)
            st.session_state.doc_names = st.session_state.store.list_documents()
            if not st.session_state.doc_names:
                st.session_state.store = None
            reset_results()
            st.rerun()
    else:
        st.caption('No documents indexed yet.')

    with st.expander('Session controls'):
        if st.button('Clear chat', disabled=not st.session_state.chat_history, width='stretch'):
            st.session_state.chat_history = []
            st.session_state.last_sources = []
            st.session_state.last_elapsed = 0.0
            st.rerun()
        if st.button('Clear documents', disabled=st.session_state.store is None, width='stretch'):
            st.session_state.store = None
            st.session_state.doc_names = []
            reset_results()
            st.rerun()
        st.caption('Session data is temporary; a refresh can reset it.')
    with st.expander('Retrieval and diagnostics'):
        st.caption('Dense / FAISS finds related meaning. BM25 matches terms. Hybrid RRF combines ranks; it is not always better.')
        st.caption(f"Current retrieval: {'Hybrid RRF' if USE_HYBRID_SEARCH else 'Dense / FAISS'}")
        st.caption(f'LLM: {GROK_MODEL} · Embeddings: {EMBEDDING_MODEL_NAME}')
        judge_enabled = st.toggle('Score answers with an LLM judge', value=False,
                                  help='Optional extra provider request after each answer. Scores are estimates, not guarantees.')

if st.session_state.store is None:
    demo_card, upload_card = st.columns(2, gap='medium')
    with demo_card, st.container(border=True):
        st.subheader('Try Demo')
        st.write('Explore two fictional battery and fleet reports. No documents to prepare.')
        if st.button('Try Demo', type='primary', width='stretch', icon=':material/play_arrow:'):
            from fixture_utils import write_fixture_pdfs
            import tempfile
            runtime = ROOT / '.runtime' / 'demo'
            runtime.mkdir(parents=True, exist_ok=True)
            try:
                with processing('Preparing demo documents…'):
                    with tempfile.TemporaryDirectory(dir=runtime) as directory:
                        st.session_state.store = build_index_from_pdfs(write_fixture_pdfs(directory), persist=False)
                        st.session_state.doc_names = st.session_state.store.list_documents()
                        reset_results()
                        st.session_state.demo_active = True
                st.rerun()
            except (RuntimeError, ValueError, OSError):
                st.error('Could not load the demo. Check that the embedding model is available.')
        st.caption('All demo facts are fictional.')
    with upload_card, st.container(border=True):
        st.subheader('Upload PDFs')
        st.write('Add one or more PDFs using the sidebar, then choose Build / Update Index.')
        st.caption('Native text extraction with optional OCR. Each document stays in your session.')
    st.caption('Selected PDF text is sent to the configured provider only for answers, summaries, comparisons, or optional judging.')
    st.stop()

st.caption(f"{len(st.session_state.doc_names)} documents ready · Use document selection to keep ambiguous questions in scope.")
tab_chat, tab_search, tab_summary, tab_cross, tab_eval = st.tabs(['Chat / Q&A', 'Evidence Search', 'Summarize', 'Cross-Document', 'Evaluation'])

suggested_query = None
with tab_chat:
    st.subheader('Ask your documents')
    if len(st.session_state.doc_names) > 1:
        st.multiselect('Document selection · leave empty to search all', st.session_state.doc_names, key='chat_scope',
                       help='Select documents when a question could refer to more than one project.')
    if not st.session_state.chat_history:
        st.caption('Ask a question below. Inspect page references and retrieved evidence alongside the answer.')
    if st.session_state.demo_active:
        st.markdown('**Try asking**')
        suggestions = (
            'How much energy does the Aurora battery project store?',
            'What are the limitations of the Aurora battery system?',
            'What is MF4 telemetry used for?',
            'Compare the maintenance intervals of Aurora and Meridian.',
        )
        suggestion_columns = st.columns(2)
        for index, question in enumerate(suggestions):
            with suggestion_columns[index % 2]:
                if st.button(question, key=f'demo_question_{index}', width='stretch'):
                    suggested_query = question
    for message in st.session_state.chat_history:
        with st.chat_message(message['role']):
            if message.get('error'):
                st.error(message['content'])
            else:
                render_with_latex(message['content'])
            if message['role'] == 'assistant':
                show_citation_status(message.get('citation_validation'))
                if message.get('sources'):
                    with st.expander(f"Retrieved evidence ({len(message['sources'])})", expanded=False):
                        show_sources(message['sources'])
                if message.get('elapsed') is not None:
                    st.caption(f"Answered in {message['elapsed']:.1f}s")
    chat_turn = st.container()
    if st.session_state.chat_history:
        transcript = '\n\n'.join(f"### {m['role'].title()}\n\n{m['content']}" for m in st.session_state.chat_history)
        st.download_button('Download conversation', transcript, 'docquery-conversation.md', mime='text/markdown')

with tab_search:
    st.subheader('Evidence Search')
    strategy = 'Hybrid RRF' if USE_HYBRID_SEARCH else 'Dense / FAISS'
    st.caption(f'Retrieval strategy: {strategy}. Scores order evidence; they are not answer-confidence probabilities.')
    with st.form('search_form'):
        search_query = st.text_input('Search query', key='search_box')
        top_k = st.slider('Results to show', 1, 15, 5)
        search_scope = st.multiselect('Search documents', st.session_state.doc_names, key='search_scope')
        submitted = st.form_submit_button('Search')
    if submitted and search_query.strip():
        try:
            with processing('Searching your documents…'):
                st.session_state.search_results = retrieve(search_query, st.session_state.store, top_k=top_k, doc_filter=search_scope or None)
        except (RuntimeError, ValueError, OSError):
            st.error('Search failed. Check that the embedding model is available.')
    if st.session_state.search_results is not None:
        results = st.session_state.search_results
        if not results:
            st.info('No indexed evidence matches the selected scope.')
        show_sources([{'doc_name': r['chunk'].doc_name, 'page': r['chunk'].page_number, 'text': r['chunk'].text,
                       'source': r['chunk'].source, 'chunk_id': r['chunk'].chunk_id,
                       'score': r['score'], 'score_type': r.get('score_type', 'rrf')} for r in results])

with tab_summary:
    st.subheader('Summarize a document')
    st.caption('Generate a concise brief across the document, with page citations to inspect.')
    selected_doc = st.selectbox('Choose a document', st.session_state.doc_names)
    if st.button('Generate summary'):
        try:
            with processing('Generating document summary…'):
                summary = summarize_document(selected_doc, st.session_state.store)
                st.session_state.summary_result = (selected_doc, summary)
        except RuntimeError as exc:
            st.error(str(exc))
    if st.session_state.summary_result:
        name, summary = st.session_state.summary_result
        with st.container(border=True):
            st.markdown(f'**Summary · {name}**')
            render_with_latex(summary)
            refs = [{'doc_name': c.doc_name, 'page': c.page_number, 'chunk_id': c.chunk_id}
                    for c in st.session_state.store.chunks if c.doc_name == name]
            show_citation_status(validate_citations(summary, refs))
        st.download_button('Download summary', summary, 'docquery-summary.md', mime='text/markdown')

with tab_cross:
    st.subheader('Cross-Document')
    st.caption('Compare or combine retrieved evidence across your indexed documents. Inspect which pages were selected.')
    cross_query = st.text_input('Cross-document question', key='cross_box')
    if st.button('Compare documents', disabled=not cross_query.strip()):
        try:
            with processing('Comparing evidence across documents…'):
                st.session_state.cross_result = cross_document_query(cross_query, st.session_state.store)
        except RuntimeError as exc:
            st.error(str(exc))
    if st.session_state.cross_result:
        result = st.session_state.cross_result
        with st.container(border=True):
            render_with_latex(result['answer'])
            show_citation_status(result.get('citation_validation'))
            st.caption(f"Documents retrieved: {', '.join(result['documents_used'])}")
            with st.expander('Retrieved evidence', expanded=False):
                show_sources(result['sources'])

with tab_eval:
    st.subheader('Frozen offline benchmark')
    try:
        benchmark, questions, documents = frozen_benchmark()
        st.caption(f'{questions} questions · {documents} fictional documents')
        display = benchmark.copy()
        for column in ('Hit@3', 'Recall@3', 'MRR@3'):
            display[column] = display[column].map(lambda value: f'{value:.3f}')
        st.table(display.set_index('Method'))
        best = benchmark.loc[benchmark['MRR@3'].idxmax(), 'Method']
        st.write(f'**{best} performed best on this controlled benchmark.**')
        st.caption('This benchmark is a controlled project fixture and does not establish general real-world performance. Source: docs/evaluation.md.')
    except (OSError, ValueError):
        st.warning('The frozen benchmark could not be loaded. See docs/evaluation.md.')
    st.subheader('Current-session diagnostics')
    st.caption('These describe the evidence and timings of your chat answers, separately from the offline benchmark.')
    df = logs_to_dataframe(st.session_state.eval_logs)
    if df.empty:
        st.caption('No chat queries logged yet. Ask a question to inspect this session; no quality scoring runs automatically.')
    else:
        st.caption(f"{len(df)} queries logged · Rank scores describe retrieval ordering, not answer confidence.")
        columns = [c for c in ('query', 'num_results', 'latency_sec', 'answer_latency_sec', 'citation_status') if c in df]
        labels = {'query': 'Question', 'num_results': 'Evidence chunks', 'latency_sec': 'Retrieval (s)',
                  'answer_latency_sec': 'Answer (s)', 'citation_status': 'Citation references'}
        st.dataframe(df[columns].rename(columns=labels), hide_index=True, width='stretch')
        quality = df[['groundedness', 'relevance']].apply(pd.to_numeric, errors='coerce').dropna(how='all')
        if quality.empty:
            st.caption('LLM quality judging has not produced scores for this session. Enable it in sidebar diagnostics only if you want an additional provider request.')
        else:
            st.caption('Optional LLM judge estimates · 1–5 · Not factual verification')
            st.dataframe(quality.rename(columns={'groundedness': 'Groundedness', 'relevance': 'Relevance'}), width='stretch')
        if len(df) >= 2:
            with st.expander('Session trends', expanded=False):
                times = [c for c in ('latency_sec', 'answer_latency_sec') if c in df]
                fig = px.line(df, x=df.index, y=times, title='Local session timings', labels={'x': 'Query', 'value': 'Seconds'})
                st.plotly_chart(fig, width='stretch')
                if len(quality) >= 2:
                    fig = px.line(quality, x=quality.index, y=['groundedness', 'relevance'], title='Optional judge estimates', labels={'x': 'Query', 'value': 'Score (1–5)'})
                    st.plotly_chart(fig, width='stretch')
        with st.expander('Detailed session log', expanded=False):
            st.dataframe(df, hide_index=True, width='stretch')
        st.download_button('Download evaluation log', df.to_csv(index=False), 'docquery-evaluation.csv', mime='text/csv')
        if st.button('Clear evaluation log'):
            st.session_state.eval_logs = []
            st.rerun()

# Keep input and its temporary processing status pinned to the page bottom.
chat_query = st.chat_input('Ask a question about your documents…', submit_mode='disable')
chat_query = chat_query or suggested_query
if chat_query:
    st.session_state.chat_history.append({'role': 'user', 'content': chat_query})
    scope = st.session_state.get('chat_scope', [])
    with chat_turn, st.chat_message('user'):
        render_with_latex(chat_query)
    try:
        with processing('Searching your documents and generating an answer…', pinned=True) as status:
            start = time.perf_counter()
            result = answer_question(chat_query, st.session_state.store,
                                     chat_history=st.session_state.chat_history[:-1], doc_filter=scope or None)
            elapsed = time.perf_counter() - start
            st.session_state.chat_history.append({'role': 'assistant', 'content': result['answer'],
                                                 'sources': result['sources'], 'citation_validation': result.get('citation_validation'),
                                                 'elapsed': elapsed})
            st.session_state.last_sources = result['sources']
            st.session_state.last_elapsed = elapsed
            if result['sources']:
                if judge_enabled:
                    status.update(label='Scoring answer quality…')
                    judge = evaluate_answer_quality(chat_query, result['sources'], result['answer'])
                else:
                    judge = {'groundedness': None, 'relevance': None, 'judge_status': 'not_requested'}
                retr = summarize_retrieval(chat_query, result['sources'], result.get('retrieval_seconds', 0))
                st.session_state.eval_logs.append({**retr, **judge, 'answer_latency_sec': round(elapsed, 3),
                                                  'citation_status': result.get('citation_validation', {}).get('status'),
                                                  'scope': scope, 'retrieval_query': result.get('retrieval_query', chat_query)})
    except RuntimeError as exc:
        st.session_state.chat_history.append({'role': 'assistant', 'content': str(exc), 'error': True})
        st.session_state.last_sources = []
    st.rerun()
