library(ggmap)

df <- read.csv('../../data/data_train_competition.csv')

df <- df[c('starting_latitude', 'starting_longitude', 'revenue_class')]
names(df) <- c('lat', 'lon', 'y')

map. <- get_map('Thessaloniki', zoom=10)
map <- ggmap(map., extent='device', legend='topleft')


df1 <- df[df$y == 1,]
df3 <- df[df$y == 3,]
df5 <- df[df$y == 5,]
p <- map
p <- p + stat_density2d(aes(lon, lat, alpha=..level..), df5, fill='red', geom='polygon')
p <- p + stat_density2d(aes(lon, lat, alpha=..level..), df3, fill='green', geom='polygon')
p <- p + stat_density2d(aes(lon, lat, alpha=..level..), df1, fill='blue', geom='polygon')
p <- p + scale_alpha_continuous(range=c(0.1, 0.6))
print(p)

p <- map + geom_point(aes(lon, lat, color=y), df, size=1)
p <- p + scale_colour_gradient(name='revenue', limits=c(1, 5), low='white', high='red')
print(p)
ggsave('map.png', p)

