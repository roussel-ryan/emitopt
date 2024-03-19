def centroid_position(input_dict, trim_name_x, trim_name_y, quad_name, init_xpos=1., init_ypos=1.): 
    trim_x = input_dict[trim_name_x]
    trim_y = input_dict[trim_name_y]
    quad = input_dict[quad_name]

    xpos = init_xpos + trim_x + quad*(init_xpos + trim_x)
    ypos = init_ypos + trim_y - quad*(init_ypos + trim_y) # note the negative sign for opposite kick in y-direction
    result = {'xpos': float(xpos)*100,
               'ypos': float(ypos)*100}
    return result