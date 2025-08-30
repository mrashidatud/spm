# main.py ------------------------------------------------------
from model    import write_csvs
from plotting import plots

def main():
    df = write_csvs()
    # plots(df)                       # full data
    # for cut in (100, 20):       # filtered convenience plots
    #     plots(df[df.total < cut], thresh=cut)
    # print("Pipeline complete")

if __name__ == "__main__":
    main()
