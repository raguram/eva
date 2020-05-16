def persistToZip(images, zip, outFolderName, fileNameStart=0, extn="jpg"):
    names = []
    for i, img in enumerate(images):
        img.save(f"temp.{extn}")
        name = f"{outFolderName}/Image_{fileNameStart + i}.{extn}"
        zip.write("temp.jpg", name)
        names.append(name)
    return names