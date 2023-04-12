# isthisloss
 An attempt to build a CNN that can tell if an image is the Internet meme *loss*.
 loss can take on many forms. It's initially derived from a page from the webcomic Ctrl Alt Del, but has since been abstracted into a very simple visual pattern:

 |  ||

 || |_

 Any image that follows this pattern or in some other way directly references the original webcomic page can be considered "loss".

 A primary challenge is to find instances of images that are not loss. While this is true of most images, it's not necessarily trivial to find a collection of images that are most suited to training this network. For instance, memes that reference *other* pages of Ctrl Alt Del are not-loss; the training data must ensure that the CNN does not straightforwardly identify all Ctrl Alt Del comics as loss. 

 To my knowledge, there are four types of memes that can be considered loss, in increasing order of abstraction. These can serve as a guide for what we should include in the training data as examples of not-loss:
 * Memes that directly use parts of the original Ctrl Alt Del comic. To select for these, we need a set of other Ctrl Alt Del pages in the training data.
 * Memes that follow a 4-panel format similar to the original comic but does not actually use art from the original comic. To select for these, we need include a set of 4 page comics that are not loss.
 * Memes that are do not follow a 4-panel format but nonetheless show the abstracted pattern. Maybe a standard CV image dataset would work?
 * Memes that are simple text-like, geometric patterns very similar to the basic abstract pattern. 

 This repo includes several side components:
 scrapers: Some scripts to scrape the Internet for the desired images.
 panel-counter: A simple CNN trained on "comics" generated from Tiny Imagenet images to find 4-panel comics.