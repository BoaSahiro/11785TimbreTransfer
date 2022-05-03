from pytube import Playlist

playlist = Playlist('https://www.youtube.com/watch?v=oS_yNsDsbaQ&list=PLFxfNcRmI5YpNrKGThakSLJmGiZ22zSNa')
print('Number of videos in playlist: %s' % len(playlist.video_urls))

# Loop through all videos in the playlist and download them
for video in playlist.videos:
    stream = video.streams.get_highest_resolution()
    stream.download("guitar")
    # print("hey")
    # video.streams.download()



# from pytube import YouTube, Playlist

# # importing packages
# from pytube import YouTube
# import os

# def download(yt_link):
#     # print(yt_link)
#     # url input from user
#     yt = YouTube(yt_link)
    
#     # extract only audio
#     video = yt.streams.filter(only_audio=True).first()
    
#     # check for destination to save file
#     destination = "trumpet"
    
#     # download the file
#     out_file = video.download(output_path=destination)
    
#     # # save the file
#     base, ext = os.path.splitext(out_file)
#     base = base.replace(" ", "") # remove all white spaces
#     new_file = base + '.wav'
#     os.rename(out_file, new_file)
    
#     # result of success
#     print(yt.title + " has been successfully downloaded.")