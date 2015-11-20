function numGoodProposals = visualizeFilteredProposals( proposals, img, innerbounds, outerbounds )

    figure;
    imshow(img)
    hold on
    
    numGoodProposals = 0;
    for i = 1:size(proposals,1)
        bbox = proposals(i,:);
        if containsBBox(bbox, innerbounds) && containsBBox(outerbounds, bbox)
            x = bbox(1);
            y = bbox(2);
            w = bbox(3) - x;
            h = bbox(4) - y;

            rectangle('Position', [x y w h], 'EdgeColor', 'r');
            numGoodProposals = numGoodProposals + 1;
        end
    end
    hold off

end

function fits = containsBBox( outer, inner )

    fits = true;
    if outer(1) > inner(1)
        fits = false;
    elseif outer(2) > inner(2)
        fits = false;
    elseif outer(3) < inner(3)
        fits = false;
    elseif outer(4) < inner(4)
        fits = false;
    end

end
