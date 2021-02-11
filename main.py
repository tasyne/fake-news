# Initial exploratory analysis by Leland Ball, Feb 2021
# conda install pandas
# conda install -c conda-forge pandas-profiling
# conda install -c conda-forge arrow

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path
import arrow

TRUE_CSV = 'True.csv'
FAKE_CSV = 'Fake.csv'
CORPUS_DIR = 'fake_real_news'
PROFILE_REPORT_HTML = 'fake_news_profile.html'

def date_from_string(s):
    try:
        return arrow.get(s.strip(), 'MMMM D, YYYY')  # 'December 16, 2020' format
    except arrow.parser.ParserMatchError:
        pass

    try:
        return arrow.get(s.strip(), 'D-MMM-YY')  # '19-Feb-18' format
    except arrow.parser.ParserMatchError:
        pass

    try:
        return arrow.get(s.strip(), 'MMM D, YYYY')  # 'Dec 31, 2017' format
    except arrow.parser.ParserMatchError:
        pass

    #raise Exception('SuperParseMatchError', f"Couldn't parse date: {s}")
    #print(f"Couldn't parse date: {s}")
    return None

def import_data(path_true_news, path_fake_news):
    path_true_news, path_fake_news = Path(path_true_news), Path(path_fake_news)
    # csv to df
    pd_true, pd_false = pd.read_csv(path_true_news), pd.read_csv(path_fake_news)

    # add labels to all
    pd_true['label'] = 1
    pd_false['label'] = 0

    # combine
    df_all = pd.concat([pd_true, pd_false])

    # dates from string to datetime objects
    df_all['date'] = df_all['date'].apply(date_from_string)
    # Drop unparseable (there are only 10 of them)
    df_all = df_all.dropna(subset=['date'])

    return df_all


def vectorize_content(df, label_col='label', text_col='text'):
    # maneuver columns around to be as desired: ["label", "other columns", ...]
    cols = list(df.columns)
    cols.remove(label_col)
    df = df[[label_col] + cols]

    # get frequencies using CountVectorizer
    # preserve punctuation as tokens, because those are probably "important?"
    # e.g.: Betteridge's law of headlines
    cv = CountVectorizer(
        input='content',
        stop_words='english',
        token_pattern=r"(?u)\b\w\w+\b|!|\?|\"|\'")
    w = cv.fit_transform(list(df[text_col]))

    # convert to DataFrame without label
    df_vectors = pd.DataFrame(w.A, columns=cv.get_feature_names())

    # uppercase columns that aren't from the df_vectors dataframe
    df.columns = map(str.upper, df.columns)

    # add labels (combine both data frames)
    df = df.reset_index()
    df = pd.concat([df[list(df.columns)], df_vectors], axis=1)
    return df


def main(dir_main, make_profile=False):
    dir_main = Path(dir_main)
    path_true_news = Path(dir_main / CORPUS_DIR / TRUE_CSV)
    path_fake_news = Path(dir_main / CORPUS_DIR / FAKE_CSV)
    path_profile = Path(dir_main / PROFILE_REPORT_HTML)

    # load and format data
    df_all = import_data(path_true_news, path_fake_news)

    if make_profile:
        from pandas_profiling import ProfileReport  # takes forever
        prof = ProfileReport(df_all)
        prof.to_file(output_file=path_profile)

    # vectorize title only
    df_all = vectorize_content(df_all, label_col='label', text_col='title')

    df_all.describe()

    pass


if __name__ == '__main__':
    dir_main = Path.cwd()
    main(dir_main, make_profile=True)
