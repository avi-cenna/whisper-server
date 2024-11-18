def main [] {
    let list = [1 2 3 4 5 6 7 8 9 10]
    # print $list
    $list |  par-each { |it| 
        let url = $"http://localhost:8000/items/($it)"
        http get $url
    }
}
